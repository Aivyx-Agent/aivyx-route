//! Optional small-model classifier: which tier does free-form chat need?
//! The crate supplies the prompt and a strict parser; the product makes
//! the call (with a grammar and timeout) and treats `None` as `Medium`.

use std::fmt::Write as _;

use crate::profile::Tier;

/// Per-message cap (in chars) so the classifier prompt stays small.
pub const MAX_MESSAGE_CHARS: usize = 500;

const INSTRUCTIONS: &str = "\
Classify how capable a language model must be to answer the latest request well.
Answer with exactly one word: small, medium, or large.
- small: greetings, short factual lookups, simple rewording or formatting.
- medium: ordinary questions, drafting, summaries, single-step tasks.
- large: multi-step reasoning, planning, code, analysis, or high-stakes ambiguity.

Conversation (oldest first):
";

/// Build the classifier prompt from the most recent messages, oldest first.
pub fn prompt(recent: &[&str]) -> String {
    let mut out = String::from(INSTRUCTIONS);
    for (i, message) in recent.iter().enumerate() {
        let clipped: String = message.chars().take(MAX_MESSAGE_CHARS).collect();
        // Writing to a String cannot fail.
        let _ = writeln!(out, "{}. {clipped}", i + 1);
    }
    out.push_str("\nAnswer:");
    out
}

/// Parse the classifier's answer. Anything but one exact tier word is `None`.
pub fn parse(output: &str) -> Option<Tier> {
    match output.trim().to_ascii_lowercase().as_str() {
        "small" => Some(Tier::Small),
        "medium" => Some(Tier::Medium),
        "large" => Some(Tier::Large),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_exact_tier_words() {
        assert_eq!(parse("small"), Some(Tier::Small));
        assert_eq!(parse(" Large\n"), Some(Tier::Large));
        assert_eq!(parse("MEDIUM"), Some(Tier::Medium));
    }

    #[test]
    fn parse_rejects_anything_else() {
        for bad in [
            "",
            "large.",
            "big",
            "small medium",
            "tier: small",
            "medium-ish",
        ] {
            assert_eq!(parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn prompt_lists_messages_in_order_and_names_every_tier() {
        let p = prompt(&["hi there", "now plan my week"]);
        assert!(p.contains("small") && p.contains("medium") && p.contains("large"));
        let first = p.find("1. hi there").unwrap();
        let second = p.find("2. now plan my week").unwrap();
        assert!(first < second);
        assert!(p.ends_with("\nAnswer:"));
    }

    #[test]
    fn prompt_truncates_long_messages() {
        let long = "é".repeat(MAX_MESSAGE_CHARS + 50);
        let p = prompt(&[&long]);
        assert!(p.contains(&"é".repeat(MAX_MESSAGE_CHARS)));
        assert!(!p.contains(&"é".repeat(MAX_MESSAGE_CHARS + 1)));
    }
}

//! A per-session map with a size cap, for state a long-running product
//! keeps per conversation (sticky model, pin, last decision). Conversations
//! come and go without telling the router, so without a cap these maps
//! would only grow. At the cap, inserting a new session evicts the one
//! updated least recently — an idle conversation just starts fresh.

use std::collections::HashMap;

/// How many sessions a [`SessionMap`] keeps by default.
pub const MAX_SESSIONS: usize = 1024;

#[derive(Debug, Clone)]
pub struct SessionMap<V> {
    cap: usize,
    /// Monotonic counter: an entry's stamp is when it was last inserted.
    clock: u64,
    entries: HashMap<String, (V, u64)>,
}

impl<V> Default for SessionMap<V> {
    fn default() -> Self {
        SessionMap::with_capacity(MAX_SESSIONS)
    }
}

impl<V> SessionMap<V> {
    /// A map keeping at most `cap` sessions (at least one).
    pub fn with_capacity(cap: usize) -> Self {
        SessionMap {
            cap: cap.max(1),
            clock: 0,
            entries: HashMap::new(),
        }
    }

    pub fn get(&self, session: &str) -> Option<&V> {
        self.entries.get(session).map(|(value, _)| value)
    }

    /// Sets `session`'s value, marking it the most recently updated. A new
    /// session at the cap evicts the least recently updated one.
    pub fn insert(&mut self, session: String, value: V) {
        if !self.entries.contains_key(&session) && self.entries.len() >= self.cap {
            // O(cap), but only when a new session arrives at the cap.
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, (_, stamp))| *stamp)
                .map(|(key, _)| key.clone());
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
        self.clock += 1;
        self.entries.insert(session, (value, self.clock));
    }

    pub fn remove(&mut self, session: &str) -> Option<V> {
        self.entries.remove(session).map(|(value, _)| value)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_session_at_the_cap_evicts_the_least_recently_updated() {
        let mut m = SessionMap::with_capacity(2);
        m.insert("a".into(), 1);
        m.insert("b".into(), 2);
        m.insert("c".into(), 3);
        assert_eq!(m.len(), 2);
        assert_eq!(m.get("a"), None);
        assert_eq!(m.get("b"), Some(&2));
        assert_eq!(m.get("c"), Some(&3));
    }

    #[test]
    fn updating_a_session_keeps_it_and_makes_it_recent() {
        let mut m = SessionMap::with_capacity(2);
        m.insert("a".into(), 1);
        m.insert("b".into(), 2);
        // Updating an existing session never evicts…
        m.insert("a".into(), 10);
        assert_eq!(m.len(), 2);
        assert_eq!(m.get("b"), Some(&2));
        // …and makes it the most recent, so `b` goes next.
        m.insert("c".into(), 3);
        assert_eq!(m.get("a"), Some(&10));
        assert_eq!(m.get("b"), None);
    }

    #[test]
    fn remove_and_a_zero_cap() {
        let mut m = SessionMap::with_capacity(0);
        m.insert("a".into(), 1);
        assert_eq!(m.get("a"), Some(&1), "a cap of 0 still keeps one");
        assert_eq!(m.remove("a"), Some(1));
        assert!(m.is_empty());
        assert_eq!(m.remove("a"), None);
    }

    #[test]
    fn the_default_cap_is_max_sessions() {
        let mut m = SessionMap::default();
        for i in 0..=MAX_SESSIONS {
            m.insert(i.to_string(), i);
        }
        assert_eq!(m.len(), MAX_SESSIONS);
        assert_eq!(m.get("0"), None);
    }
}

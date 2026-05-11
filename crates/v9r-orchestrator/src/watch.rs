//! Path-pattern routing for the orchestrator.
//!
//! Two patterns are needed:
//! - **wildcard** (`/agents/*/manifest.toml`): matches any agent dir
//! - **exact** (`/agents/echo/input`): registered per-agent at manifest time
//!
//! When several routes match the same path, the most-specific (most literal
//! segments) wins. This makes a per-agent exact route override a generic
//! wildcard if both are registered for the same path.

use std::sync::Arc;

use v9r_core::{VfsError, VfsPath, VfsResult};

use crate::TriggerHandler;

#[derive(Clone, Debug)]
pub struct PathPattern {
    segments: Vec<PatternSegment>,
}

#[derive(Clone, Debug)]
enum PatternSegment {
    Literal(String),
    Wildcard,
}

impl PathPattern {
    /// `*` is one segment, no slashes. Partial wildcards (`fo*o`) are
    /// rejected — the pattern language stays trivially auditable.
    pub fn parse(s: &str) -> VfsResult<Self> {
        let mut segments = Vec::new();
        for seg in s.split('/') {
            if seg.is_empty() {
                continue;
            }
            if seg == "*" {
                segments.push(PatternSegment::Wildcard);
            } else if seg.contains('*') {
                return Err(VfsError::InvalidPath(format!(
                    "partial wildcards not supported: {seg}"
                )));
            } else {
                segments.push(PatternSegment::Literal(seg.to_string()));
            }
        }
        Ok(Self { segments })
    }

    pub fn exact(path: &VfsPath) -> Self {
        Self {
            segments: path
                .segments()
                .iter()
                .map(|s| PatternSegment::Literal(s.clone()))
                .collect(),
        }
    }

    pub fn matches(&self, path: &VfsPath) -> bool {
        let s = path.segments();
        if s.len() != self.segments.len() {
            return false;
        }
        self.segments.iter().zip(s).all(|(p, seg)| match p {
            PatternSegment::Literal(l) => l == seg,
            PatternSegment::Wildcard => true,
        })
    }

    pub fn specificity(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| matches!(s, PatternSegment::Literal(_)))
            .count()
    }
}

#[derive(Clone)]
pub struct Route {
    pub label: String,
    pub pattern: PathPattern,
    pub handler: Arc<dyn TriggerHandler>,
}

#[derive(Default)]
pub struct WatchTable {
    routes: Vec<Route>,
}

impl WatchTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a route. If a route with the same label already exists, it is
    /// replaced (upsert). Labels are how `ManifestHandler` swaps an agent's
    /// routes atomically when its manifest is rewritten.
    pub fn register(
        &mut self,
        label: impl Into<String>,
        pattern: PathPattern,
        handler: Arc<dyn TriggerHandler>,
    ) {
        let label = label.into();
        self.routes.retain(|r| r.label != label);
        self.routes.push(Route {
            label,
            pattern,
            handler,
        });
    }

    pub fn unregister(&mut self, label: &str) -> bool {
        let before = self.routes.len();
        self.routes.retain(|r| r.label != label);
        before != self.routes.len()
    }

    /// Drop every route whose label starts with `prefix`. Used to clear
    /// out an agent's routes (`agent:<name>:*`) before re-registering them.
    pub fn unregister_prefix(&mut self, prefix: &str) -> usize {
        let before = self.routes.len();
        self.routes.retain(|r| !r.label.starts_with(prefix));
        before - self.routes.len()
    }

    pub fn match_path(&self, path: &VfsPath) -> Option<Arc<dyn TriggerHandler>> {
        self.routes
            .iter()
            .filter(|r| r.pattern.matches(path))
            .max_by_key(|r| r.pattern.specificity())
            .map(|r| r.handler.clone())
    }

    pub fn labels(&self) -> Vec<String> {
        self.routes.iter().map(|r| r.label.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use v9r_runtime::AgentRuntime;
    use v9r_vfs::{Vfs, WriteEvent};

    struct DummyHandler;
    #[async_trait]
    impl TriggerHandler for DummyHandler {
        async fn handle(
            &self,
            _: WriteEvent,
            _: Arc<Vfs>,
            _: Arc<dyn AgentRuntime>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn ptr_id(h: &Arc<dyn TriggerHandler>) -> usize {
        Arc::as_ptr(h) as *const () as usize
    }

    #[test]
    fn pattern_wildcard_matches() {
        let p = PathPattern::parse("/agents/*/manifest.toml").unwrap();
        assert!(p.matches(&VfsPath::parse("/agents/echo/manifest.toml").unwrap()));
        assert!(p.matches(&VfsPath::parse("/agents/foo/manifest.toml").unwrap()));
        assert!(!p.matches(&VfsPath::parse("/agents/echo/input").unwrap()));
        assert!(!p.matches(&VfsPath::parse("/agents/manifest.toml").unwrap()));
    }

    #[test]
    fn pattern_exact_matches() {
        let p = PathPattern::exact(&VfsPath::parse("/agents/echo/input").unwrap());
        assert!(p.matches(&VfsPath::parse("/agents/echo/input").unwrap()));
        assert!(!p.matches(&VfsPath::parse("/agents/foo/input").unwrap()));
    }

    #[test]
    fn most_specific_wins() {
        let mut wt = WatchTable::new();
        let exact: Arc<dyn TriggerHandler> = Arc::new(DummyHandler);
        let wild: Arc<dyn TriggerHandler> = Arc::new(DummyHandler);
        let exact_id = ptr_id(&exact);

        wt.register("wild", PathPattern::parse("/agents/*/input").unwrap(), wild);
        wt.register(
            "exact",
            PathPattern::exact(&VfsPath::parse("/agents/echo/input").unwrap()),
            exact,
        );

        let h = wt
            .match_path(&VfsPath::parse("/agents/echo/input").unwrap())
            .unwrap();
        assert_eq!(ptr_id(&h), exact_id);
    }

    #[test]
    fn register_replaces_same_label() {
        let mut wt = WatchTable::new();
        wt.register(
            "x",
            PathPattern::parse("/a").unwrap(),
            Arc::new(DummyHandler),
        );
        wt.register(
            "x",
            PathPattern::parse("/b").unwrap(),
            Arc::new(DummyHandler),
        );
        assert_eq!(wt.labels(), vec!["x".to_string()]);
        assert!(wt.match_path(&VfsPath::parse("/a").unwrap()).is_none());
        assert!(wt.match_path(&VfsPath::parse("/b").unwrap()).is_some());
    }

    #[test]
    fn unregister_prefix() {
        let mut wt = WatchTable::new();
        wt.register(
            "agent:echo:input",
            PathPattern::parse("/agents/echo/input").unwrap(),
            Arc::new(DummyHandler),
        );
        wt.register(
            "agent:echo:ctl",
            PathPattern::parse("/agents/echo/ctl").unwrap(),
            Arc::new(DummyHandler),
        );
        wt.register(
            "manifest",
            PathPattern::parse("/agents/*/manifest.toml").unwrap(),
            Arc::new(DummyHandler),
        );
        let removed = wt.unregister_prefix("agent:echo:");
        assert_eq!(removed, 2);
        assert_eq!(wt.labels(), vec!["manifest".to_string()]);
    }
}

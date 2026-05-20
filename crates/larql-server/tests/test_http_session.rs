#![cfg(not(target_arch = "wasm32"))]
//! HTTP integration tests: SessionManager tests.

mod common;
use common::*;

use larql_server::session::SessionManager;

// ══════════════════════════════════════════════════════════════
// ASYNC STATE / SESSION MANAGER TESTS
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn session_manager_list_empty_for_unknown_session() {
    let sm = SessionManager::new(3600);
    let patches = sm.list_patches("session-xyz").await;
    assert!(patches.is_empty());
}

#[tokio::test]
async fn session_manager_apply_patch_and_list() {
    let sm = SessionManager::new(3600);
    let m = model("test");

    // After REV3, apply_patch is safe to call directly on a never-seen
    // session_id — the slow path reads `model.patched` outside the
    // sessions write guard.

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-26".into(),
        description: Some("my-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    };

    let (op_count, active) = sm.apply_patch("sess-1", &m, patch).await;
    assert_eq!(op_count, 1);
    assert_eq!(active, 1);

    let list = sm.list_patches("sess-1").await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "my-patch");
}

#[tokio::test]
async fn session_manager_remove_nonexistent_patch_returns_err() {
    let sm = SessionManager::new(3600);
    let m = model("test");
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-26".into(),
        description: Some("my-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    };
    sm.apply_patch("sess-1", &m, patch).await;

    let err = sm.remove_patch("sess-1", "nonexistent").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn session_manager_remove_patch_by_name() {
    let sm = SessionManager::new(3600);
    let m = model("test");

    for name in &["patch-a", "patch-b"] {
        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-04-26".into(),
            description: Some((*name).into()),
            author: None,
            tags: vec![],
            operations: vec![larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 1,
                reason: None,
            }],
        };
        sm.apply_patch("sess-2", &m, patch).await;
    }

    let remaining = sm.remove_patch("sess-2", "patch-a").await.unwrap();
    assert_eq!(remaining, 1);

    let list = sm.list_patches("sess-2").await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "patch-b");
}

#[tokio::test]
async fn session_manager_remove_from_unknown_session_returns_err() {
    let sm = SessionManager::new(3600);
    let err = sm.remove_patch("no-such-session", "any-patch").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("not found"));
}

// ══════════════════════════════════════════════════════════════
// REV3 — apply_patch must not block the runtime worker on the
// slow path (`model.patched.blocking_read()` was the bug).
// ══════════════════════════════════════════════════════════════

fn one_op_patch(name: &str) -> larql_vindex::VindexPatch {
    larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-05-10".into(),
        description: Some(name.into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_patch_slow_path_makes_progress_with_held_patched_reader() {
    // Holds `model.patched.read()` for the entire duration. Pre-REV3,
    // apply_patch's `or_insert_with` did `model.patched.blocking_read()`
    // while holding `sessions.write().await` on a worker thread —
    // *another* concurrent task that needed `patched.write()` would
    // deadlock against it. Read-while-read is safe (RwLock allows
    // multiple readers), so this test asserts forward progress under
    // the realistic concurrent-read load that exercises the slow path.
    let sm = std::sync::Arc::new(SessionManager::new(3600));
    let m = model("test");

    let m_for_holder = m.clone();
    let holder = tokio::spawn(async move {
        let _guard = m_for_holder.patched.read().await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    });

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sm.apply_patch("never-seen", &m, one_op_patch("first")),
    )
    .await
    .expect("apply_patch must finish within 5s — REV3 deadlock canary");

    assert_eq!(result, (1, 1));
    holder.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_apply_patch_same_session_finishes() {
    // Hammers the same session_id from many tasks. Pre-REV3, the slow
    // path serialised on a `blocking_read` while every other task held
    // `sessions.write()` — observable as worker-thread starvation. Post-
    // REV3 the slow path drops the sessions guard around the patched
    // read, and a race is absorbed via `entry().or_insert_with`.
    let sm = std::sync::Arc::new(SessionManager::new(3600));
    let m = model("test");

    let mut tasks = Vec::new();
    for i in 0..16 {
        let sm = sm.clone();
        let m = m.clone();
        tasks.push(tokio::spawn(async move {
            sm.apply_patch("contended", &m, one_op_patch(&format!("p{i}")))
                .await
        }));
    }

    let outcomes = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let mut out = Vec::new();
        for t in tasks {
            out.push(t.await.unwrap());
        }
        out
    })
    .await
    .expect("16 concurrent apply_patch calls must finish within 5s");

    assert_eq!(outcomes.len(), 16);
    // Every call applies one operation; final active count should equal 16.
    let final_active = outcomes.iter().map(|(_, n)| *n).max().unwrap();
    assert_eq!(final_active, 16);
    let list = sm.list_patches("contended").await;
    assert_eq!(list.len(), 16);
}

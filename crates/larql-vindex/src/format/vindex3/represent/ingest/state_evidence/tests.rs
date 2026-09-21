use super::*;

#[test]
fn evidence_starts_with_only_a_path_and_preserves_the_verified_binding() {
    let f = super::super::tests::Fixture::new();
    let evidence = ArtifactStateEvidence::establish(&f.candidate).unwrap();
    assert_eq!(
        evidence.established(),
        f.prepared.request().unwrap().key().state()
    );
    assert_eq!(evidence.candidate_authority_digest().len(), 64);
    let moved = tempfile::tempdir().unwrap();
    let destination = moved.path().join("candidate");
    std::fs::rename(&f.candidate, &destination).unwrap();
    drop(f);
    assert_eq!(
        ArtifactStateEvidence::establish(&destination).unwrap(),
        evidence
    );
}

#[test]
fn incomplete_candidate_authority_still_refuses() {
    let dir = tempfile::tempdir().unwrap();
    assert!(ArtifactStateEvidence::establish(dir.path()).is_err());
}

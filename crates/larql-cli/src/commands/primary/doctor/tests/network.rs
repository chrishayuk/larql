use super::super::network::{
    hub_checks, hub_endpoint, hub_probe_url, parse_semver, release_checks, HubReply, Probe,
    DEFAULT_HF_ENDPOINT, LATEST_RELEASE_PAGE,
};
use super::super::report::Status;
use super::only;

#[test]
fn the_endpoint_honours_hf_endpoint_unless_blank() {
    assert_eq!(hub_endpoint(None), DEFAULT_HF_ENDPOINT);
    assert_eq!(hub_endpoint(Some("  ".into())), DEFAULT_HF_ENDPOINT);
    assert_eq!(
        hub_endpoint(Some("https://mirror.example/".into())),
        "https://mirror.example"
    );
}

#[test]
fn the_probe_asks_for_the_quick_start_model() {
    assert_eq!(
        hub_probe_url("https://huggingface.co"),
        "https://huggingface.co/api/models/Qwen/Qwen3-0.6B"
    );
}

#[test]
fn semver_accepts_release_tags_and_refuses_the_rest() {
    assert_eq!(parse_semver("v0.2.1"), Some((0, 2, 1)));
    assert_eq!(parse_semver("1.10.0"), Some((1, 10, 0)));
    assert_eq!(parse_semver("vindex-v0.8.0"), None);
    assert_eq!(parse_semver("0.2"), None);
    assert_eq!(parse_semver("0.2.1.4"), None);
    assert_eq!(parse_semver("0.2.1-rc1"), None);
}

#[test]
fn hub_outcomes() {
    let endpoint = DEFAULT_HF_ENDPOINT;
    let ok = only(hub_checks(
        endpoint,
        &Probe::Done(HubReply {
            status: 200,
            millis: 9,
        }),
    ));
    assert_eq!(ok.status, Status::Ok);
    assert_eq!(ok.name, "huggingface.co");
    assert!(ok.summary.contains("9 ms"));

    let odd = only(hub_checks(
        endpoint,
        &Probe::Done(HubReply {
            status: 503,
            millis: 9,
        }),
    ));
    assert_eq!(odd.status, Status::Warn);
    assert!(odd.summary.contains("503"));

    let down = only(hub_checks(endpoint, &Probe::Failed("timed out".into())));
    assert_eq!(down.status, Status::Fail);
    assert!(down.hint.unwrap().contains("--offline"));

    assert_eq!(
        only(hub_checks(endpoint, &Probe::Skipped)).status,
        Status::Skip
    );
}

#[test]
fn an_endpoint_without_a_scheme_is_named_whole() {
    assert_eq!(only(hub_checks("mirror", &Probe::Skipped)).name, "mirror");
}

#[test]
fn release_outcomes() {
    let newer = only(release_checks("0.2.1", &Probe::Done("v0.3.0".into())));
    assert_eq!(newer.status, Status::Warn);
    assert!(newer.hint.unwrap().contains(LATEST_RELEASE_PAGE));

    for tag in ["v0.2.1", "v0.2.0"] {
        let same = only(release_checks("0.2.1", &Probe::Done(tag.into())));
        assert_eq!(same.status, Status::Ok, "{tag}");
    }

    let odd = only(release_checks(
        "0.2.1",
        &Probe::Done("vindex-v0.8.0".into()),
    ));
    assert_eq!(odd.status, Status::Skip);

    let failed = only(release_checks(
        "0.2.1",
        &Probe::Failed("rate limited".into()),
    ));
    assert_eq!(failed.status, Status::Skip);
    assert!(failed.summary.contains("rate limited"));

    let skipped = only(release_checks("0.2.1", &Probe::Skipped));
    assert_eq!(skipped.summary, "skipped (--offline)");
}

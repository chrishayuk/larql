//! `serve_one` against the synthetic speech model: the request-to-reply
//! path a resident server runs, without a checkpoint.

use super::super::*;
use super::scratch;
use larql_inference::speech::test_support::{
    synth_moss_tokenizer, synth_speech_model, SynthSpeechModel, SYNTH_RVQ,
};

/// Frame cap for every request here: enough to see frames, small enough
/// to keep the synthetic run quick.
const TEST_MAX_FRAMES: usize = 4;
/// Text made of words the synthetic tokenizer knows.
const TEST_TEXT: &str = "hello world one two three";

fn args(voice: Option<PathBuf>) -> SpeakServeArgs {
    SpeakServeArgs {
        model: String::new(),
        listen: None,
        voice,
        max_frames: TEST_MAX_FRAMES,
        seed: 0,
        greedy: false,
        q4: false,
        metal: false,
    }
}

/// Serve one request against a freshly built synthetic model.
fn serve(args: &SpeakServeArgs, request: Request) -> Result<String, BoxErr> {
    let SynthSpeechModel {
        backbone,
        depth,
        audio_tables,
        config,
    } = synth_speech_model();
    let resident = Resident {
        config,
        tokenizer: synth_moss_tokenizer(),
        backend: larql_inference::cpu_engine_backend(),
        weights: &backbone,
        audio_tables,
        depth,
    };
    let ffn = WeightFfn {
        weights: resident.weights,
    };
    let depth_ffn = WeightFfn {
        weights: &resident.depth.weights,
    };
    serve_one(&resident, args, &ffn, &depth_ffn, None, None, request)
}

fn request(text: &str, tokens: Option<PathBuf>) -> Request {
    Request {
        text: Some(text.to_string()),
        tokens,
        ..Request::default()
    }
}

/// A voice reference in token-row form, every id inside the audio vocab.
fn write_voice(path: &Path) {
    let row: Vec<String> = (0..SYNTH_RVQ).map(|id| id.to_string()).collect();
    let line = row.join(" ");
    std::fs::write(path, format!("# voice\n{line}\n{line}\n")).expect("write voice");
}

fn token_rows(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .expect("tokens file")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .count()
}

#[test]
fn serve_one_writes_tokens_and_reports_ok() {
    let dir = scratch("serve_tokens");
    let tokens = dir.join("out.tokens.txt");
    let reply = serve(&args(None), request(TEST_TEXT, Some(tokens.clone()))).expect("served");
    assert!(reply.starts_with("ok frames="), "{reply}");
    assert!(reply.contains(" tokens="), "{reply}");
    let rows = token_rows(&tokens);
    assert!((1..=TEST_MAX_FRAMES).contains(&rows), "{rows} rows");
}

#[test]
fn serve_one_rejects_a_request_without_text() {
    let error = serve(&args(None), request("   ", None)).expect_err("no text");
    assert!(error.to_string().contains("no text"), "{error}");
}

#[test]
fn serve_one_rejects_a_request_with_nowhere_to_put_frames() {
    let error = serve(&args(None), request(TEST_TEXT, None)).expect_err("no sink");
    assert!(error.to_string().contains("tokens:"), "{error}");
}

/// The server default voice is used when the request names none, and a
/// missing default is a request error rather than a crash.
#[test]
fn serve_one_inherits_the_default_voice() {
    let dir = scratch("serve_default_voice");
    let voice = dir.join("voice.tokens");
    write_voice(&voice);
    let tokens = dir.join("out.tokens.txt");
    serve(&args(Some(voice)), request(TEST_TEXT, Some(tokens))).expect("served");

    let missing = dir.join("missing.tokens");
    serve(
        &args(Some(missing)),
        request(TEST_TEXT, Some(dir.join("x.txt"))),
    )
    .expect_err("missing default voice");
}

/// `voice: -` drops a default voice that would otherwise fail to load.
#[test]
fn serve_one_unconditioned_sentinel_skips_the_default_voice() {
    let dir = scratch("serve_unconditioned");
    let mut request = request(TEST_TEXT, Some(dir.join("out.tokens.txt")));
    request.voice = Some(UNCONDITIONED.to_string());
    serve(&args(Some(dir.join("missing.tokens"))), request).expect("unconditioned");
}

#[test]
fn serve_one_takes_a_per_request_voice() {
    let dir = scratch("serve_request_voice");
    let voice = dir.join("voice.tokens");
    write_voice(&voice);
    let mut request = request(TEST_TEXT, Some(dir.join("out.tokens.txt")));
    request.voice = Some(voice.display().to_string());
    request.greedy = Some(true);
    request.seed = Some(7);
    request.max_frames = Some(2);
    serve(&args(None), request).expect("served");
}

/// `stream:` gets one flushed line per frame; with no `tokens:` the reply
/// names no token file.
#[test]
fn serve_one_streams_every_frame() {
    let dir = scratch("serve_stream");
    let stream = dir.join("frames.txt");
    std::fs::write(&stream, "").expect("stream sink");
    let mut request = request(TEST_TEXT, None);
    request.stream = Some(stream.clone());
    let reply = serve(&args(None), request).expect("served");
    assert!(!reply.contains(" tokens="), "{reply}");
    let streamed = std::fs::read_to_string(&stream).expect("stream");
    let frames: usize = reply
        .split_whitespace()
        .find_map(|field| field.strip_prefix("frames="))
        .and_then(|count| count.parse().ok())
        .expect("frames= in reply");
    // Every generated frame is streamed; EOS may add one past `emitted`.
    assert!(streamed.lines().count() >= frames, "{streamed:?}");
    for line in streamed.lines() {
        assert_eq!(line.split_whitespace().count(), SYNTH_RVQ, "{line:?}");
    }
}

#[test]
fn serve_one_reports_a_stream_it_cannot_open() {
    let dir = scratch("serve_stream_missing");
    let mut request = request(TEST_TEXT, None);
    request.stream = Some(dir.join("absent").join("frames.fifo"));
    serve(&args(None), request).expect_err("stream open fails");
}

#[test]
fn run_refuses_a_directory_without_config_json() {
    let dir = scratch("run_no_config");
    let mut args = args(None);
    args.model = dir.display().to_string();
    let error = run(args).expect_err("no config.json");
    assert!(error.to_string().contains("config.json"), "{error}");
}

#[test]
fn run_reports_an_unloadable_checkpoint() {
    let dir = scratch("run_bad_checkpoint");
    std::fs::write(dir.join("config.json"), "{}").expect("config");
    let mut args = args(None);
    args.model = dir.display().to_string();
    run(args).expect_err("no weights to load");
}

#[test]
fn write_frame_line_writes_one_row() {
    let mut out = Vec::new();
    write_frame_line(&mut out, &[1, 2, 3]).expect("write");
    assert_eq!(out, b"1 2 3\n");
}

#[test]
fn respond_without_a_reply_path_goes_to_stdout() {
    respond(None, "ok frames=1").expect("stdout");
}

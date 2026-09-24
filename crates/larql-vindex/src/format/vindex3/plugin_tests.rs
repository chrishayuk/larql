//! Tests for the plugin ABI stamp and registrar. In their own file: the
//! lowering-closure gate forbids constructing a provider in `plugin.rs`.

use super::*;

#[test]
fn abi_is_terminated_and_names_its_parts() {
    assert!(ABI.ends_with('\0'));
    assert_eq!(ABI.matches('\0').count(), 1);
    let abi = abi();
    assert!(abi.starts_with("larql-plugin/1 larql-vindex/"), "{abi}");
    assert!(abi.contains("rustc"), "{abi}");
    assert!(abi.contains(" commit "), "{abi}");
}

fn reference_lowering() -> Box<dyn PlanBackend + Send> {
    Box::new(crate::format::vindex3::opplan::exec::reference::ReferenceBackend::new())
}

/// A registrar hands back exactly what the plugin registered, in order.
#[test]
fn a_registrar_returns_its_registrations_in_order() {
    use crate::format::vindex3::represent::codec::codecs::mxfp4::Mxfp4Codec;
    use crate::format::vindex3::represent::codec::codecs::nvfp4::Nvfp4Codec;
    use crate::format::vindex3::represent::codec::encoder::tests::RawF32Codec;
    let mut registrar = PluginRegistrar::new();
    registrar.codec(Box::new(Nvfp4Codec));
    registrar.codec(Box::new(Mxfp4Codec));
    registrar.encoder(Box::new(RawF32Codec));
    registrar.lowering(reference_lowering);
    let PluginRegistrations {
        codecs,
        encoders,
        lowerings,
    } = registrar.into_parts();
    let labels: Vec<&str> = codecs.iter().map(|c| c.encoding_label()).collect();
    assert_eq!(
        labels,
        [Nvfp4Codec.encoding_label(), Mxfp4Codec.encoding_label()]
    );
    let encoder_labels: Vec<&str> = encoders.iter().map(|e| e.encoding_label()).collect();
    assert_eq!(encoder_labels, [RawF32Codec.encoding_label()]);
    assert_eq!(lowerings.len(), 1);
    assert_eq!(
        lowerings[0]().identity(),
        reference_lowering().identity(),
        "the factory builds the provider it was registered as"
    );
    let none = PluginRegistrar::default().into_parts();
    assert!(none.codecs.is_empty() && none.encoders.is_empty() && none.lowerings.is_empty());
}

#[test]
fn only_the_hosts_own_stamp_is_compatible() {
    assert_eq!(abi_compatible(abi()), !abi().ends_with("commit unknown"));
    assert!(!abi_compatible(""));
    assert!(!abi_compatible(&format!("{} ", abi())));
    assert!(!abi_compatible(
        "larql-plugin/1 larql-vindex/0.0.0 (rustc x) commit unknown"
    ));
}

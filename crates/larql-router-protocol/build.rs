fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("PROTOC", protobuf_src::protoc());
    tonic_build::compile_protos("proto/grid.proto")?;
    tonic_build::compile_protos("proto/expert.proto")?;
    // attention-service-routes change.
    tonic_build::compile_protos("proto/attention.proto")?;
    Ok(())
}

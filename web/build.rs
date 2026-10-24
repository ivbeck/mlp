// rust-embed needs the folder to exist, a fresh checkout has no frontend build yet
fn main() {
    let dist = std::path::Path::new("ui/dist");
    if !dist.exists() {
        let _ = std::fs::create_dir_all(dist);
    }
    println!("cargo:rerun-if-changed=ui/dist");
}

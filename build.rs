fn main() {
    cc::Build::new()
        .file("extern/galois/galois.c")
        .warnings(false)
        .compile("galois");
}

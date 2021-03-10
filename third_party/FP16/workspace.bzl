"""Loads the FP16 library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "FP16",
        strip_prefix = "FP16-4dfe081cf6bcd15db339cf2680b9281b8451eeb3",
        sha256 = "90f20492621d5ed80b442aa682ff92d7ccf333ac8fac4a10e7e02afb159f3c13",
        urls = [
            "https://mirror.bazel.build/github.com/Maratyszcza/FP16/archive/4dfe081cf6bcd15db339cf2680b9281b8451eeb3.tar.gz",
            "https://github.com/Maratyszcza/FP16/archive/4dfe081cf6bcd15db339cf2680b9281b8451eeb3.tar.gz",
        ],
        build_file = "//third_party/FP16:BUILD.bazel",
    )

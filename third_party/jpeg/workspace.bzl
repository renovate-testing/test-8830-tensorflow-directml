"""loads the jpeg library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "jpeg",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.90.tar.gz",
            "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.90.tar.gz",
        ],
        sha256 = "6a965adb02ad898b2ae48214244618fe342baea79db97157fdc70d8844ac6f09",
        strip_prefix = "libjpeg-turbo-2.0.90",
        build_file = "//third_party/jpeg:BUILD.bazel",
        system_build_file = "//third_party/jpeg:BUILD.system",
    )

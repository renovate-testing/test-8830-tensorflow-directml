"""loads the highwayhash library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "highwayhash",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/highwayhash/archive/bdd572de8cfa3a1fbef6ba32307c2629db7c4773.tar.gz",
            "https://github.com/google/highwayhash/archive/bdd572de8cfa3a1fbef6ba32307c2629db7c4773.tar.gz",
        ],
        sha256 = "ebe6fe710f2a4a9417410773a1aeb8e750b036d4843e457054e6209c1fd0b043",
        strip_prefix = "highwayhash-bdd572de8cfa3a1fbef6ba32307c2629db7c4773",
        build_file = "//third_party/highwayhash:BUILD.bazel",
    )

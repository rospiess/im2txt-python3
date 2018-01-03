package(default_visibility = ["//im2txt:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "evaluate_im2txt",
    srcs = ["evaluate_im2txt.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//im2txt/evaludation/pycocoevalcap:eval",
        "//im2txt/evaludation/pycocotools:coco",
    ],
)
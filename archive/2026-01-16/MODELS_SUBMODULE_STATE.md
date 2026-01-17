# Models submodule state snapshot

git status --porcelain models:
 M models/LFM2.5-VL-1.6B

git diff -- models/LFM2.5-VL-1.6B:
diff --git a/models/LFM2.5-VL-1.6B b/models/LFM2.5-VL-1.6B
--- a/models/LFM2.5-VL-1.6B
+++ b/models/LFM2.5-VL-1.6B
@@ -1 +1 @@
-Subproject commit 9747fc86ed7a39e37e397fa9ee00b003fb0ade42
+Subproject commit 9747fc86ed7a39e37e397fa9ee00b003fb0ade42-dirty

ls -la models/LFM2.5-VL-1.6B:
total 3123184
drwxrwxr-x 4 sophia sophia       4096 Jan 14 15:55 .
drwxrwxr-x 3 sophia sophia       4096 Jan 15 08:10 ..
drwxrwxr-x 3 sophia sophia       4096 Jan 14 15:41 .cache
-rw-rw-r-- 1 sophia sophia       2217 Jan 14 15:41 chat_template.jinja
-rw-rw-r-- 1 sophia sophia       2376 Jan 14 15:41 config.json
-rw-rw-r-- 1 sophia sophia        136 Jan 14 15:42 generation_config.json
drwxrwxr-x 8 sophia sophia       4096 Jan 16 19:00 .git
-rw-rw-r-- 1 sophia sophia       1519 Jan 14 15:42 .gitattributes
-rw-rw-r-- 1 sophia sophia      10596 Jan 14 15:41 LICENSE
-rw-rw-r-- 1 sophia sophia 3193334216 Jan 14 15:55 model.safetensors
-rw-rw-r-- 1 sophia sophia        828 Jan 14 15:41 processor_config.json
-rw-rw-r-- 1 sophia sophia      11203 Jan 14 15:41 README.md
-rw-rw-r-- 1 sophia sophia        843 Jan 14 15:42 tokenizer_config.json
-rw-rw-r-- 1 sophia sophia    4733040 Jan 14 15:42 tokenizer.json

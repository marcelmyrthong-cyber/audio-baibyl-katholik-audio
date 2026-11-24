# Ka Audio Baibyl Katholik: recordings

Chapter recordings for the Ka Audio Baibyl Katholik app. They are published as files on the
[**audio** release](../../releases/tag/audio), one MP3 per chapter, named `<bookId>_<chapter>.mp3`
(for example `gen_1.mp3`, `psa_23.mp3`, `mat_5.mp3`). The app reads that release to find recordings,
streams them, and lets listeners download them for offline use.

To publish recordings, push them into the `audio/` folder of a branch named `audio-inbox`.
The *Publish audio* workflow uploads them to the release and deletes the branch, so the
recordings don't fill up this repository's history.

Recordings © the parish. All rights reserved.

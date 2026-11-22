# Demo sessions

Pre-computed Group Pulse summaries for four AMI Meeting Corpus meetings (Edinburgh room), shown in the public demo without any upload or processing. Each file is the output of `python -m grouppulse.social.report --tracks <meeting>.csv --out <meeting>.json`: derived numbers only (attention, engagement, spells, 10 s windows). No video, audio or faces.

| File | Meeting | Note |
| --- | --- | --- |
| ES2002a.json | Remote control design, kickoff | gaze model never saw these people |
| ES2002b.json | Remote control design, functional | same group as ES2002a |
| ES2003a.json | Remote control design, kickoff (group 2) | |
| ES2008a.json | Remote control design, kickoff (group 3) | the Edinburgh gaze model was calibrated on this meeting, so its gaze targets are in-sample |

Loaded into the backend at start-up (`python -m grouppulse.api.import_ami demo/sessions`) and bundled into the static demo build (`scripts/export_demo.py`).

Data credit: AMI Meeting Corpus, CC BY 4.0. Carletta, J. et al. (2005), *The AMI Meeting Corpus: A Pre-announcement*. https://groups.inf.ed.ac.uk/ami/corpus/

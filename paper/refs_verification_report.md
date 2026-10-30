# Reference verification report

Paper: MLSys 2027 submission. Verification date: 2026-10-30.

Method: for each key, the bibliographic record was pulled from Crossref (DOI metadata deposited by the publisher) and cross-checked against the publisher or proceedings page (ACM DL, IEEE Xplore via Crossref, JMLR, ACL Anthology, USENIX, NeurIPS proceedings, BMVA archive, PNAS, Elsevier), the arXiv API, and dblp where it was reachable. Author lists are complete; no entry uses "et al.". Where Crossref returned an IEEE DOI in lower case, the DOI is written in the publisher's canonical upper-case form (DOIs are case-insensitive).

Summary: 25 of 25 batch keys VERIFIED, plus bridges2016understanding verified separately (26) and two entries carried over from the program's published MLSP 2026 bibliography (27). One key (nvidia_nvml, unused in the paper) is verified as an online resource but the page exposes no version number, which is stated in its note field. One key (yang2023parttime) has a retitled peer-reviewed successor; both are provided and both are cited.

Output files:
- paper/refs.bib (29 entries)
- paper/refs_verification_report.md (this file)

---

## 1. horowitz2014 - VERIFIED

- Source: Crossref record for DOI 10.1109/ISSCC.2014.6757323 (https://doi.org/10.1109/ISSCC.2014.6757323); first page of the paper PDF (https://www.gwern.net/doc/cs/hardware/2014-horowitz-2.pdf) confirms title, session 1.1, page 10, and ISBN 978-1-4799-0920-9.
- Confirmed: title "1.1 Computing's energy problem (and what we can do about it)", Mark Horowitz, pages 10-14, February 2014, San Francisco.
- Correction: the full proceedings title is "2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers (ISSCC)". DOI added.

## 2. williams2009roofline - VERIFIED

- Source: Crossref record for DOI 10.1145/1498765.1498785 (https://doi.org/10.1145/1498765.1498785); Zenodo mirror https://zenodo.org/record/1236156.
- Confirmed: Samuel Williams, Andrew Waterman, David Patterson; Communications of the ACM 52(4), pages 65-76, April 2009.
- Correction: none. DOI added.

## 3. choi2013roofline - VERIFIED

- Source: Crossref record for DOI 10.1109/IPDPS.2013.77 (https://doi.org/10.1109/IPDPS.2013.77).
- Confirmed: Jee Whan Choi, Daniel Bedard, Robert Fowler, Richard Vuduc; "A Roofline Model of Energy"; 2013 IEEE 27th International Symposium on Parallel and Distributed Processing (IPDPS), Cambridge, MA, May 2013.
- Correction: pages 661-672 and DOI added. Note that a 2012 Georgia Tech technical report of the same title lists Choi as sole author (https://repository.gatech.edu/entities/publication/1277da76-1ca2-49ab-8bfe-fcea87f4f780); the IPDPS paper is the four-author version and is the one cited.

## 4. henderson2020systematic - VERIFIED

- Source: JMLR abstract page and BibTeX, https://jmlr.org/papers/v21/20-312.html and https://jmlr.org/papers/v21/20-312.bib.
- Confirmed: Peter Henderson, Jieru Hu, Joshua Romoff, Emma Brunskill, Dan Jurafsky, Joelle Pineau; JMLR 21(248), 2020.
- Correction: pages 1-43 added. JMLR does not assign DOIs to this volume, so no DOI field.

## 5. strubell2019energy - VERIFIED

- Source: ACL Anthology BibTeX, https://aclanthology.org/P19-1355.bib (landing page https://aclanthology.org/P19-1355/).
- Confirmed: Emma Strubell, Ananya Ganesh, Andrew McCallum; Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Florence, Italy, July 2019.
- Correction: pages 3645-3650, DOI 10.18653/v1/P19-1355, and editors (Korhonen, Traum, Marquez) added.

## 6. schwartz2020green - VERIFIED

- Source: Crossref record for DOI 10.1145/3381831 (https://doi.org/10.1145/3381831).
- Confirmed: Roy Schwartz, Jesse Dodge, Noah A. Smith, Oren Etzioni; "Green AI"; Communications of the ACM 63(12), December 2020.
- Correction: pages 54-63 and DOI added.

## 7. patterson2021carbon - VERIFIED

- Source: arXiv abstract page https://arxiv.org/abs/2104.10350 and the arXiv API record (v3, 23 April 2021).
- Confirmed: "Carbon Emissions and Large Neural Network Training", arXiv 2104.10350, 2021. Complete author list (9): David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, Jeff Dean.
- Correction: full author list supplied in place of "et al.". No journal or conference version exists on arXiv or Crossref; cited as a preprint with the arXiv DataCite DOI 10.48550/arXiv.2104.10350 (resolves).

## 8. burtscher2014measuring - VERIFIED

- Source: Crossref record for DOI 10.1145/2588768.2576783 (https://doi.org/10.1145/2588768.2576783); author's PDF https://userweb.cs.txstate.edu/~mb92/papers/gpgpu14.pdf (first page shows GPGPU-7, March 1 2014, Salt Lake City, ACM 978-1-4503-2766-4).
- Confirmed: Martin Burtscher, Ivan Zecena, Ziliang Zong; "Measuring GPU Power with the K20 Built-in Sensor"; Proceedings of Workshop on General Purpose Processing Using GPUs (GPGPU-7), ACM, March 2014.
- Correction: pages 28-36, ISBN, and DOI added. Caution: Crossref holds two DOIs for this paper, 10.1145/2588768.2576783 (created 2014, 48 citations, resolves to the ACM DL landing page) and 10.1145/2576779.2576783 (created 2020, 18 citations). The 2014-created DOI whose proceedings prefix 10.1145/2588768 is the registered GPGPU-7 proceedings DOI is used. The ACM DL itself returned HTTP 403 to automated fetches, so the landing page was not inspected directly.

## 9. yang2023parttime - VERIFIED (preprint), with published successor identified

- Source: arXiv abstract page https://arxiv.org/abs/2312.02741 and arXiv API record; Crossref record for DOI 10.1109/SC41406.2024.00028; OpenAlex record for that DOI.
- Confirmed: "Part-time Power Measurements: nvidia-smi's Lack of Attention", Zeyu Yang, Karel Adamek, Wesley Armour, arXiv 2312.02741, first posted 5 December 2023 (v3 dated 12 December 2024). Primary category cs.DC.
- Later version found: the arXiv record carries DOI 10.1109/SC41406.2024.00028, which Crossref and OpenAlex resolve to a retitled peer-reviewed paper by the same three authors: "Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU's Built-In Power Sensor", in SC24: International Conference for High Performance Computing, Networking, Storage and Analysis, Atlanta, GA, November 2024, pages 1-17, IEEE. This is a conference paper, not a journal article. It is included in the .bib as a separate entry, yang2024accurate, so the paper can cite either the preprint title or the published title. IEEE Xplore returned an empty page to automated fetching, so the published title and pages rest on Crossref and OpenAlex metadata (both agree).
- Spelling: the second author appears as "Karel Adamek" on arXiv and with an accent in the paper body and in OpenAlex; the accented form is used in the .bib.

## 10. you2023zeus - VERIFIED

- Source: USENIX presentation page with BibTeX, https://www.usenix.org/conference/nsdi23/presentation/you; arXiv 2208.06102 (comment field "NSDI 2023").
- Confirmed: Jie You, Jae-Won Chung, Mosharaf Chowdhury; "Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training"; 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), Boston, MA, April 2023.
- Correction: pages 119-139 and ISBN 978-1-939133-33-5 added. USENIX assigns no DOI.

## 11. denton2014exploiting - VERIFIED

- Source: NeurIPS proceedings BibTeX and metadata JSON, https://proceedings.neurips.cc/paper_files/paper/2014/file/1adaeb993eba95859121a43ea61bd858-Bibtex.bib and ...-Metadata.json (page_first 1269, page_last 1277); dblp record conf/nips/DentonZBLF14 (pages 1269-1277); volume page https://proceedings.neurips.cc/paper/2014 (editors Ghahramani, Welling, Cortes, Lawrence, Weinberger).
- Confirmed: Emily Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, Rob Fergus; Advances in Neural Information Processing Systems 27 (NIPS 2014), Curran Associates.
- Correction: pages 1269-1277, editors, and publisher added. The older URL hash (2afe4567...) that search engines still index returns 404; the current canonical page hash is 1adaeb99..., which is what the .bib uses. The arXiv version (1404.0736) lists the first author as "Remi Denton"; the proceedings form "Emily Denton" is used.

## 12. jaderberg2014speeding - VERIFIED

- Source: BMVA archive BibTeX, https://bmva-archive.org.uk/bmvc/2014/files/paper073.bib; Crossref record for DOI 10.5244/C.28.88.
- Confirmed: Max Jaderberg, Andrea Vedaldi, Andrew Zisserman; Proceedings of the British Machine Vision Conference 2014, BMVA Press, Nottingham.
- Correction: pages 88.1-88.13, DOI 10.5244/C.28.88, and editors (Valstar, French, Pridmore) added.

## 13. hu2022lora - VERIFIED

- Source: ICLR 2022 virtual site https://iclr.cc/virtual/2022/poster/6319 (lists the OpenReview forum nZeVKeeFYf9); ML Anthology record https://mlanthology.org/iclr/2022/hu2022iclr-lora/; arXiv API record for 2106.09685 (8 authors).
- Confirmed: "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022 (poster). Complete author list (8): Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.
- Correction: full author list supplied in place of "et al.". OpenReview itself served a browser-verification page to automated fetching, so the OpenReview URL is taken from the ICLR site. ICLR proceedings have no pages or DOI.

## 14. sainath2013lowrank - VERIFIED

- Source: Crossref record for DOI 10.1109/ICASSP.2013.6638949 (https://doi.org/10.1109/ICASSP.2013.6638949).
- Confirmed: Tara N. Sainath, Brian Kingsbury, Vikas Sindhwani, Ebru Arisoy, Bhuvana Ramabhadran; 2013 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Vancouver, May 2013.
- Correction: publisher capitalisation of the title is "Low-rank matrix factorization for Deep Neural Network training with high-dimensional output targets". Pages 6655-6659 and DOI added.

## 15. nosek2018preregistration - VERIFIED

- Source: Crossref record for DOI 10.1073/pnas.1708274114 (https://doi.org/10.1073/pnas.1708274114).
- Confirmed: Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, David T. Mellor; "The preregistration revolution"; PNAS 115(11), March 2018.
- Correction: pages 2600-2606 and DOI added.

## 16. lacoste2019quantifying - VERIFIED

- Source: arXiv abstract page https://arxiv.org/abs/1910.09700 and arXiv API record (v2, 4 November 2019).
- Confirmed: Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, Thomas Dandres; "Quantifying the Carbon Emissions of Machine Learning"; arXiv 1910.09700, 2019. Primary category cs.CY.
- Correction: none. No journal or conference version found; cited as a preprint with the arXiv DataCite DOI 10.48550/arXiv.1910.09700 (resolves).

## 17. luccioni2023bloom - VERIFIED

- Source: JMLR BibTeX, https://jmlr.org/papers/v24/23-0069.bib (landing page https://jmlr.org/papers/v24/23-0069.html).
- Confirmed: Alexandra Sasha Luccioni, Sylvain Viguier, Anne-Laure Ligozat; JMLR 24, 2023.
- Correction: issue number 253 and pages 1-15 added. No DOI assigned by JMLR.

## 18. anthony2020carbontracker - VERIFIED

- Source: arXiv abstract page https://arxiv.org/abs/2007.03051 and arXiv API record (comment: "Accepted to be presented at the ICML Workshop on Challenges in Deploying and monitoring Machine Learning Systems, 2020"); ICML virtual site https://icml.cc/virtual/2020/7921 (workshop: Challenges in Deploying and Monitoring Machine Learning Systems).
- Confirmed: Lasse F. Wolff Anthony, Benjamin Kanding, Raghavendra Selvan; arXiv 2007.03051, July 2020; ICML 2020 workshop paper.
- Correction: workshop name made explicit in howpublished. The ICML virtual page lists only the first author; the arXiv record lists all three, which is what the .bib uses. No workshop proceedings DOI exists; the arXiv DataCite DOI 10.48550/arXiv.2007.03051 is given (resolves). Primary category cs.CY.

## 19. desislavov2023trends - VERIFIED

- Source: Crossref record for DOI 10.1016/j.suscom.2023.100857 (https://doi.org/10.1016/j.suscom.2023.100857); arXiv 2109.05472 journal-reference field (which names the same volume and article number).
- Confirmed: Radosvet Desislavov, Fernando Martinez-Plumed, Jose Hernandez-Orallo; "Trends in AI inference energy consumption: Beyond the performance-vs-parameter laws of deep learning"; Sustainable Computing: Informatics and Systems, 2023.
- Correction: volume 38, article number 100857, April 2023, DOI added. Author names carry diacritics in the published version (Martinez-Plumed, Hernandez-Orallo), encoded in LaTeX in the .bib. The earlier arXiv version had a different title ("Compute and Energy Consumption Trends in Deep Learning Inference"); the journal title is the one to cite.

## 20. nvidia_nvml - VERIFIED (online resource; version number not exposed; not cited in the paper)

- Source: https://docs.nvidia.com/deploy/nvml-api/index.html (also https://docs.nvidia.com/deploy/nvml-api/latest/index.html and .../nvml-api-reference.html). Page title "NVML API Reference Guide", publisher NVIDIA Corporation, footer "Copyright 2007-2026, NVIDIA Corporation & affiliates", "Last updated Sep 09, 2026".
- Could not confirm: the current page exposes no driver or document version string (the version metadata field is empty). Archived versions exist under https://docs.nvidia.com/deploy/archive/R5xx/nvml-api/ (for example R470, R515, R520, R535) if a pinned version is preferred; the .bib note states that no version is given.
- Cited as @misc with year 2026, month September (page date), urldate 2026-10-30.

## 21. jouppi2017tpu - VERIFIED (76 authors; not cited in the paper)

- Source: Crossref record for DOI 10.1145/3079856.3080246 (https://doi.org/10.1145/3079856.3080246), 76 authors; dblp record conf/isca/JouppiYPPABBBBB17 (76 authors, pages 1-12); arXiv 1704.04760 (75 authors).
- Confirmed: "In-Datacenter Performance Analysis of a Tensor Processing Unit"; Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA '17), Toronto, June 2017, ACM, pages 1-12.
- Correction: full 76-author list supplied in place of "et al.". The ACM/Crossref and dblp lists contain Daniel Killebrew, who is absent from the arXiv preprint author list; otherwise the lists are identical and in the same order. The published (ACM) list is used. A duplicate record exists as ACM SIGARCH Computer Architecture News 45(2), pages 1-12 (DOI 10.1145/3140659.3080246); the conference DOI is used.

## 22. chen2016eyeriss - VERIFIED (not cited in the paper)

- Source: Crossref record for DOI 10.1109/ISCA.2016.40 (https://doi.org/10.1109/ISCA.2016.40).
- Confirmed: Yu-Hsin Chen, Joel Emer, Vivienne Sze; "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks"; 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), Seoul, June 2016.
- Correction: pages 367-379 and DOI added. (A parallel record exists in ACM SIGARCH Computer Architecture News 44(3), DOI 10.1145/3007787.3001177; the IEEE conference DOI is used.)

## 23. sze2017efficient - VERIFIED

- Source: Crossref record for DOI 10.1109/JPROC.2017.2761740 (https://doi.org/10.1109/JPROC.2017.2761740).
- Confirmed: Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, Joel S. Emer; "Efficient Processing of Deep Neural Networks: A Tutorial and Survey"; Proceedings of the IEEE 105(12), December 2017.
- Correction: pages 2295-2329 and DOI added.

## 24. kandiah2021accelwattch - VERIFIED

- Source: Crossref record for DOI 10.1145/3466752.3480063 (https://doi.org/10.1145/3466752.3480063).
- Confirmed: Vijay Kandiah, Scott Peverelle, Mahmoud Khairy, Junrui Pan, Amogh Manjunath, Timothy G. Rogers, Tor M. Aamodt, Nikos Hardavellas; "AccelWattch: A Power Modeling Framework for Modern GPUs"; MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture, October 2021 (virtual event, Greece), ACM.
- Correction: pages 738-753 and DOI added.

## 25. arafa2020verified - VERIFIED (9 authors)

- Source: Crossref record for DOI 10.1145/3387902.3392613 (https://doi.org/10.1145/3387902.3392613); arXiv 2002.07795 (same DOI, same 9 authors).
- Confirmed: "Verified Instruction-Level Energy Consumption Measurement for NVIDIA GPUs"; Proceedings of the 17th ACM International Conference on Computing Frontiers (CF '20), Catania, Sicily, Italy, May 2020, ACM. Complete author list (9): Yehia Arafa, Ammar ElWazir, Abdelrahman ElKanishy, Youssef Aly, Ayatelrahman Elsayed, Abdel-Hameed Badawy, Gopinath Chennupati, Stephan Eidenbenz, Nandakishore Santhi.
- Correction: full author list supplied in place of "et al."; pages 60-70 and DOI added.

## 26. bridges2016understanding - VERIFIED (added after the batch)

- Source: Crossref record for DOI 10.1145/2962131 (https://api.crossref.org/works/10.1145/2962131).
- Confirmed: Robert A. Bridges, Neena Imam, Tiffany M. Mintz; "Understanding GPU Power: A Survey of Profiling, Modeling, and Simulation Methods"; ACM Computing Surveys 49(3), article 41, pages 1-27, published online September 2016 (issue dated 2017 by Crossref).
- Correction: the third author's given name is Tiffany (an earlier bibliography in the program had it as Todd).

## 27. syed2026toml and yang2017energy

- Carried over from the published MLSP 2026 bibliography of the same program (verified there against the FLAIRS DOI 10.32473/flairs.39.1.141781 and CVPR DOI 10.1109/CVPR.2017.643).

---

## Access limitations encountered

- dl.acm.org returned HTTP 403 to automated requests; ACM records were verified through Crossref (publisher-deposited metadata) and, for the K20 paper, the author's own PDF.
- ieeexplore.ieee.org served empty pages to automated requests; IEEE records were verified through Crossref, and for the SC24 paper additionally through OpenAlex.
- dblp.org and openreview.net served bot-check pages for most requests; two dblp records (Denton NIPS 2014, Jouppi ISCA 2017) were retrieved successfully and used.
- The Crossref, OpenAlex, and Semantic Scholar APIs rate-limited bursts of parallel requests; queries were re-run sequentially.

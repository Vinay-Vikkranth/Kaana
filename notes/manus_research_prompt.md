# Manus Research Prompt — Ka'ana Taste ID Research
### Copy everything between the START and END markers and paste into Manus

---

---START PROMPT---

## Your Mission

I am building a restaurant recommendation system for a startup called Ka'ana in Hawaii.
The CEO wants to know: **what is the minimum number of questions we need to ask a new
user to learn the maximum amount about their food and restaurant preferences?**

Your job is to find the actual academic research papers that answer this question, read
them, summarise what they say, and download the PDFs so I can verify everything myself.
I do not want summaries from blogs or news articles. I want peer-reviewed papers only.

Work through every step below in order. Do not skip steps. Do not guess or make up
citations. If you cannot find a paper, say so explicitly.

---

## STEP 1 — Set up a folder on the Desktop

Create a folder on the Desktop called: `Kaana_Research_Papers`

All downloaded PDFs go into this folder.
You will also create a summary document inside this folder at the end.

---

## STEP 2 — Search Google Scholar for the core restaurant choice papers

Go to: https://scholar.google.com

### Search 2A
Query: `restaurant selection criteria conjoint analysis attributes importance`
Filter: Any time (do not restrict by date)
Action:
- Look at the first 10 results
- Identify the 3 most-cited papers (look at the citation count shown under each result)
- For each of the 3, click on the result, find a PDF link (look for [PDF] links on the
  right side of Google Scholar, or click through to the paper page and look for a
  "Download PDF" or "Full text" button)
- Download each PDF and save to the `Kaana_Research_Papers` folder
- Name them: `GS_2A_result1.pdf`, `GS_2A_result2.pdf`, `GS_2A_result3.pdf`
- Record: title, authors, year, journal, citation count, and the URL you got it from

### Search 2B
Query: `eating out occasion restaurant choice behavior empirical`
Filter: Any time
Action: Same as 2A — find 3 most cited, download, save as `GS_2B_result1.pdf` etc.

### Search 2C
Query: `food neophobia scale preference personality eating behavior`
Filter: Any time
Action: Same — 3 most cited, download, save as `GS_2C_result1.pdf` etc.

### Search 2D
Query: `preference elicitation minimum questions recommender system cold start`
Filter: 2018–2025
Action: Same — 3 most cited, download, save as `GS_2D_result1.pdf` etc.

### Search 2E
Query: `user profiling onboarding questionnaire recommendation accuracy`
Filter: 2018–2025
Action: Same — 3 most cited, download, save as `GS_2E_result1.pdf` etc.

---

## STEP 3 — Fetch these specific papers by URL

These are papers I have been told exist. Go to each URL, confirm the paper is real,
and download the PDF.

### Paper 3A
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7503372/
Expected title: "Customer Restaurant Choice: An Empirical Analysis of Restaurant Types
and Eating-Out Occasions"
Action:
- Go to the URL
- Confirm the title matches
- Click the "PDF" button near the top of the page (PMC papers always have free PDFs)
- Download and save as `3A_Chua2020_Restaurant_Choice.pdf`
- Record: actual title, authors, year, journal as they appear on the page

### Paper 3B
URL: https://pubmed.ncbi.nlm.nih.gov/1489209/
Expected title: "Development of a scale to measure the trait of food neophobia in humans"
Expected authors: Pliner P, Hobden K (1992)
Action:
- Go to the URL
- Confirm the paper exists and the authors/title match
- Look for a "Full text links" section on the right side of the page
- Try to find a free PDF via those links
- If paywalled, search Google Scholar for `Pliner Hobden 1992 food neophobia scale`
  and find a free PDF (ResearchGate often has it)
- Download and save as `3B_Pliner_Hobden_1992_Food_Neophobia.pdf`
- Record: actual title, authors, year, journal

### Paper 3C
URL: https://arxiv.org/abs/2309.00356
Expected title: "Explainable Active Learning for Preference Elicitation"
Action:
- Go to the URL
- Confirm the paper exists
- Click "Download PDF" on the right side of the arxiv page
- Download and save as `3C_Active_Learning_Preference_Elicitation_2023.pdf`
- Record: title, authors, year

### Paper 3D
URL: https://arxiv.org/abs/2510.12015
Expected title: "Asking Clarifying Questions for Preference Elicitation With Large
Language Models"
Action:
- Go to the URL, confirm it exists
- Download PDF, save as `3D_LLM_Clarifying_Questions_2025.pdf`
- Record: title, authors, year

### Paper 3E
URL: https://www.nature.com/articles/s41598-025-09708-2
Expected title: "Active learning algorithm for alleviating the user cold start problem
of recommender systems" (2025)
Action:
- Go to the URL, confirm it exists
- Nature Scientific Reports papers are open access — download the PDF
- Save as `3E_Active_Learning_Cold_Start_2025.pdf`
- Record: title, authors, year

### Paper 3F
URL: https://arxiv.org/abs/2406.00973
Expected title: "Cold-start Recommendation by Personalized Embedding Region Elicitation"
Action:
- Go to the URL, confirm it exists
- Download PDF, save as `3F_Cold_Start_Embedding_2024.pdf`
- Record: title, authors, year

---

## STEP 4 — Search Semantic Scholar for additional validation

Go to: https://www.semanticscholar.org

### Search 4A
Query: `restaurant attribute importance customer choice`
Action:
- Look at the top 5 results
- For any paper with more than 100 citations, check if a free PDF is available
  (Semantic Scholar shows a "PDF" button when one exists)
- Download up to 3 papers, save as `SS_4A_result1.pdf` etc.
- Record title, authors, year, citation count for each

### Search 4B
Query: `cold start problem preference elicitation recommendation`
Action: Same — top 5, download those with free PDFs, up to 3 papers

---

## STEP 5 — Search PubMed for food preference and behavior papers

Go to: https://pubmed.ncbi.nlm.nih.gov

### Search 5A
Query: `food preference questionnaire validation eating behavior`
Filter: Use the "Best Match" sort
Action:
- Look at the top 5 results
- For any paper that has a "Free PMC article" or "Free article" badge, download it
- Save as `PubMed_5A_result1.pdf` etc.
- Record title, authors, year, PMID number

### Search 5B
Query: `dining occasion restaurant selection consumer behavior`
Action: Same as 5A

---

## STEP 6 — For every paper that is paywalled

If any paper in Steps 2–5 is behind a paywall (you see a page asking you to pay or
subscribe), try these fallback steps in order:

1. Copy the paper title and search Google for: `[exact paper title] filetype:pdf`
2. Check if ResearchGate has it: go to https://www.researchgate.net and search the title
3. Check if Semantic Scholar has it: go to https://www.semanticscholar.org and search
4. Check the author's university webpage (search: `[first author name] [university] publications`)
5. If none of these work, do NOT download it. Mark it as "PAYWALLED — not retrieved"
   in the summary document.

Do not create accounts on any website. Do not pay for anything.

---

## STEP 7 — Read each downloaded paper and extract key findings

For each paper you successfully downloaded, open the PDF and read enough to answer
these specific questions. Write the answers in the summary document.

For restaurant choice / selection criteria papers, answer:
- What attributes did they measure?
- Which attributes came out most important and what were their importance scores or rankings?
- What was the sample size and country?
- Did occasion / dining purpose affect the results?
- What is the single most useful finding for someone trying to design a short preference questionnaire?

For preference elicitation / recommender system papers, answer:
- What problem are they solving?
- What method do they propose for asking fewer questions?
- How many questions did they use?
- What was their measure of success?
- What is the single most useful finding for someone designing an onboarding questionnaire?

For food psychology / personality papers, answer:
- What trait or dimension are they measuring?
- What questions do they use to measure it?
- How does this trait predict food/restaurant behavior?
- What is the single most useful finding for someone designing a short preference questionnaire?

---

## STEP 8 — Create the summary document

Inside the `Kaana_Research_Papers` folder, create a document called
`RESEARCH_SUMMARY.md`

Structure it exactly like this:

```
# Ka'ana Taste ID — Research Summary
Generated by Manus on [date]

---

## Papers Successfully Downloaded

### Paper 1
- File: [filename]
- Title: [exact title from paper]
- Authors: [exact authors from paper]
- Year: [year]
- Journal/Source: [journal name or arXiv/PMC etc.]
- Citation count: [if available]
- URL where downloaded: [URL]
- Key finding for Ka'ana: [2-3 sentences in plain English]

[repeat for every downloaded paper]

---

## Papers NOT Retrieved (paywalled or not found)

### Paper X
- Title: [title I searched for]
- URL I tried: [URL]
- Reason not retrieved: [paywalled / not found / other]
- Fallback sources tried: [list what you tried]

---

## Top 5 Insights From All Papers Combined

[After reading all the papers, write the 5 most important findings that answer
the question: what are the minimum questions to ask a user to learn maximum
about their restaurant preferences? Quote specific numbers and findings from
the papers where possible.]

---

## Questions Recommended by the Research

[Based on what you read, list the specific questions the research suggests
are most important, with a citation to which paper supports each one.]
```

---

## STEP 9 — Final check

Before finishing:
1. Count how many PDFs are in the `Kaana_Research_Papers` folder
2. Make sure every PDF in the folder is listed in `RESEARCH_SUMMARY.md`
3. Make sure every paper listed in `RESEARCH_SUMMARY.md` either has a file
   or is marked as "not retrieved"
4. Open 3 random PDFs and confirm they are actual research papers (not blank,
   not error pages, not wrong documents)

Report back: "I downloaded X papers successfully. Y papers were paywalled or not
found. The summary document is complete."

---

## What I do NOT want

- Do not summarise Wikipedia articles
- Do not use blog posts, news articles, or marketing content as sources
- Do not make up citation details — if you are not sure of a detail, leave it blank
  and note that you could not verify it
- Do not download anything that requires logging in or paying
- If a URL does not exist or shows a 404 error, say so clearly — do not guess

---END PROMPT---

---

## Notes on using this prompt

- Paste everything between ---START PROMPT--- and ---END PROMPT--- into Manus
- The prompt is written so Manus can follow it step by step without any additional guidance
- Manus will create the folder, download PDFs, and write the summary automatically
- When Manus is done, open the `Kaana_Research_Papers` folder on your Desktop
  and check `RESEARCH_SUMMARY.md` to see what it found
- If any paper could not be found or was paywalled, that is useful information too —
  it means the original claim needs to be verified another way

# 🎓 EC3 Prep Hub — BITS WILP AIML Semester 1

A free, adaptive, psychologically supportive exam preparation website for BITS Pilani WILP M.Tech AIML students — covering MFML, ISM, ML, and DNN for EC3.

---

## 📁 Folder Structure

```
bits-ec3/
├── index.html                 ← Homepage (motivational landing)
├── robots.txt                 ← SEO: search engine rules
├── sitemap.xml                ← SEO: page index
├── css/
│   └── style.css              ← All styles (soft lavender/mint palette)
├── js/
│   ├── main.js                ← Nav, accordion, checklist, animations
│   └── quiz.js                ← Adaptive mock test engine (all 4 subjects)
└── pages/
    ├── syllabus.html          ← Full syllabus + textbooks
    ├── notes.html             ← Unit-wise accordion notes (all 4 subjects)
    ├── pyq.html               ← PYQ archive links + practice questions
    ├── mocktest.html          ← Adaptive quiz page
    ├── tips.html              ← Exam strategy + subject-specific tips
    ├── revision.html          ← Interactive checklist (78 topics, saved locally)
    └── confidence.html        ← Breathing exercise, quotes, anxiety support
```

---

## 🚀 GitHub Pages Deployment (Step-by-Step)

### Step 1: Create Repository
1. Go to [github.com](https://github.com) → Sign in
2. Click **"New repository"** (green button)
3. Repository name: `bits-ec3` (or any name you like)
4. Visibility: **Public** (required for free GitHub Pages)
5. Click **"Create repository"**

### Step 2: Upload Files
**Option A — GitHub Web UI (easiest, no git needed):**
1. In your new repo, click **"uploading an existing file"**
2. Drag and drop this entire `bits-ec3/` folder
3. Write commit message: `"Initial upload — EC3 Prep Hub"`
4. Click **"Commit changes"**

**Option B — Git CLI (if you have git installed):**
```bash
cd bits-ec3
git init
git add .
git commit -m "Initial upload — EC3 Prep Hub"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/bits-ec3.git
git push -u origin main
```

### Step 3: Enable GitHub Pages
1. In your repo → **Settings** tab
2. Left sidebar → **Pages**
3. Source: **Deploy from a branch**
4. Branch: **main** → Folder: **/ (root)**
5. Click **Save**
6. Wait 2–3 minutes → your site is live at:
   `https://YOURUSERNAME.github.io/bits-ec3/`

### Step 4: Update sitemap.xml
Replace `yourusername` with your actual GitHub username in:
- `sitemap.xml`
- `robots.txt`
- The `<link rel="canonical">` tag in `index.html`

---

## ✏️ How to Update Content

### Add new questions to Mock Tests:
Edit `js/quiz.js` → find `const questionBank = {` → add to the appropriate subject/level array following the existing format:
```javascript
{
  q: "Your question text here",
  options: ["Option A", "Option B", "Option C", "Option D"],
  answer: 0,  // 0-indexed correct answer
  explanation: "Why this is correct..."
}
```

### Add new checklist items (Revision page):
Edit `pages/revision.html` → find the subject's `<ul class="checklist">` → add:
```html
<li onclick="toggle(this)"><span class="check-box"></span>Your new topic</li>
```

### Update Syllabus content:
Edit `pages/syllabus.html` — all content is in plain HTML tables and cards.

---

## 📱 Optional: Custom Domain Setup
1. Buy a domain (e.g., `ec3prep.in`) from GoDaddy/Namecheap (~₹500/year)
2. In GitHub Pages Settings → Custom domain → enter your domain
3. In your domain DNS settings, add:
   ```
   A     @    185.199.108.153
   A     @    185.199.109.153
   A     @    185.199.110.153
   A     @    185.199.111.153
   CNAME www  YOURUSERNAME.github.io
   ```
4. Enable "Enforce HTTPS" in GitHub Pages settings

---

## 📣 Sharing Strategy

### WhatsApp/Telegram Groups:
Post this message in BITS WILP AIML student groups:
> "Hey all! I built a free EC3 prep site for our batch — mock tests for all 4 subjects (adaptive difficulty), unit notes, PYQ patterns, and even a revision checklist. 100% free, no login.
> 🔗 [your URL]
> Would love feedback! Share with anyone who could use it 🙏"

### What makes it shareable:
- ✅ Completely free, no ads
- ✅ Mobile friendly
- ✅ Confidence-building tone (not intimidating)
- ✅ Progress saved locally (checklist remembers your ticks)
- ✅ Adaptive quiz (encourages repeat visits)

---

## 🔑 SEO Keywords (already in meta tags)
- BITS WILP EC3 preparation
- BITS WILP AIML mock test
- MFML exam preparation
- Introduction to Statistical Methods BITS
- Machine Learning BITS WILP
- Deep Neural Networks exam prep

---

## ⚖️ Legal Precautions

1. **No copyrighted content reproduced** — All notes are original summaries
2. **PYQs** — Not hosted; we link to external community archive only
3. **Book citations** — Title + author mentioned for reference (fair use)
4. **Disclaimer** — Present in footer of every page
5. **Not affiliated** — Clearly stated on all pages

---

## 🗺️ Future Roadmap

| Phase | When | What to Add |
|-------|------|-------------|
| Phase 1 (Now) | Immediately | Deploy as-is. Share with batch. |
| Phase 2 | After EC3 | Add Semester 2 subjects. Get student feedback. |
| Phase 3 | 3–6 months | Add Google Analytics to see which pages get traffic |
| Phase 4 | 6–12 months | Consider Firebase for user progress sync (optional) |
| Phase 5 | 1+ year | Consider light monetization (Patreon/Buy Me a Coffee) only after proven value |

---

## 💜 Technical Summary

| Feature | Implementation |
|---------|---------------|
| Adaptive Quiz | JavaScript question bank, score-based level recommendation |
| Progress Tracking | localStorage (survives page refresh, private to user) |
| Breathing Exercise | CSS animation + JS phase rotation |
| Rotating Quotes | JS array with fade transitions |
| Accordion Notes | Pure CSS max-height transition |
| Mobile Nav | CSS toggle + JS class toggling |
| Checklist | Click toggle + localStorage persistence |
| Animations | IntersectionObserver for reveal on scroll |

---

Built with 💜 for the BITS WILP AIML community · 2025–2026  
100% free · No login · No ads · No tracking

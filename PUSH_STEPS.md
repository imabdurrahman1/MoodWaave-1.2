# Git Repository Initialization & Publishing Guide

This manual details the step-by-step instructions to initialize a clean, independent Git repository for the **MOODWAAVE** open-source release package, link it to a remote GitHub repository, and optionally trigger automatic DOI archival with Zenodo.

> [!WARNING]
> **DO NOT PUSH AUTOMATICALLY**: Under project ground rules, the assistant does NOT push to any remote repository. The user must review all staged files and execute the git push commands manually when ready.

---

## Pre-Flight Checklist

1. **Audio File Exclusion**: Confirm that no raw audio files (`.wav`, `.mp3`, `.au`, `.ogg`, `.flac`) are staged. The repository `.gitignore` automatically blocks these extensions.
2. **Environment Files**: Ensure virtual environment directories (`venv/`, `.venv/`) and OS caches (`__pycache__/`, `.DS_Store`, `Thumbs.db`) are excluded.
3. **Repository Directory**: These steps should be executed inside the `release/` folder if publishing MOODWAAVE as a standalone repository, or at the workspace root if publishing the entire development repository.

---

## Step-by-Step Publishing Commands

### Step 1: Open Terminal in the Release Directory

```bash
# Navigate into the release directory
cd release
```

### Step 2: Verify `.gitignore` Exists

Verify that `.gitignore` is present in the directory:

```bash
# On Linux / macOS:
ls -la .gitignore

# On Windows (PowerShell):
Get-ChildItem -Force .gitignore
```

---

### Step 3: Initialize Git Repository

Initialize a local Git repository:

```bash
git init
```

---

### Step 4: Stage All Release Files

Stage the documentation, reproduction scripts, tables, figures, models, and metadata:

```bash
git add .
```

---

### Step 5: Audit Staged Files Before Committing

Inspect the staged files to ensure no accidental binary audio files or virtual environments were staged:

```bash
# Check git status
git status

# Specifically verify no audio files are tracked (output should be empty)
git status --porcelain | grep -E "\.(wav|mp3|au|ogg|flac)$"
```

---

### Step 6: Create the Initial Release Commit

Commit all tracked files with an informative message:

```bash
git commit -m "feat: release MOODWAAVE v1.0.0 reproduction package and benchmark kit"
```

---

### Step 7: Configure Default Branch

Set the default branch to `main`:

```bash
git branch -M main
```

---

### Step 8: Connect to Remote GitHub Repository

Create an empty repository on GitHub (e.g., `https://github.com/<your-username>/MOODWAAVE`). Do not initialize it with a README, License, or `.gitignore` on GitHub.

Add the remote origin URL (choose SSH or HTTPS):

```bash
# Option A: Using SSH (recommended)
git remote add origin git@github.com:<your-username>/MOODWAAVE.git

# Option B: Using HTTPS
git remote add origin https://github.com/<your-username>/MOODWAAVE.git

# Verify remote configuration
git remote -v
```

---

### Step 9: Push to GitHub

Push your local `main` branch to GitHub and set upstream tracking:

```bash
git push -u origin main
```

---

### Step 10: Create and Push Version Tag

Tag the release for permanent academic citation and reproducibility:

```bash
git tag -a v1.0.0 -m "MOODWAAVE v1.0.0: Turnkey open-source reproduction suite"
git push origin v1.0.0
```

---

## Archival & Zenodo DOI Minting (Optional)

1. Log into [Zenodo](https://zenodo.org/) using your GitHub account.
2. Navigate to **Account Settings > GitHub**.
3. Toggle the switch next to your `MOODWAAVE` repository to **ON**.
4. On GitHub, navigate to **Releases > Create a new release**, select tag `v1.0.0`, and publish the release.
5. Zenodo will automatically ingest repository metadata from `.zenodo.json` and mint a permanent citable Digital Object Identifier (DOI).
6. Update the DOI placeholder in `CITATION.cff` and `README.md` with the minted DOI.

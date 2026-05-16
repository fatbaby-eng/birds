# Todd's Bird ID — Deployment Guide
## toddboswell.com

### What's in this ZIP

A fully self-contained static site. No server required. No database. No API keys.
The BirdNET model runs entirely in the visitor's browser — nothing is sent anywhere.

```
index.html              ← entry point
assets/
  index-*.js            ← React app (minified)
  index-*.css           ← styles (minified)
birdnet-worker.js       ← Web Worker that runs the AI model
birdnet/
  model.json            ← BirdNET model manifest
  group1-shard*.bin     ← model weights (~51 MB total)
  labels.txt            ← 6,522 species labels
```

---

### Option A — Subdirectory on toddboswell.com (recommended)

Upload the contents of this ZIP into a folder on your server, e.g. `/birds/`.

**Apache** — add a `.htaccess` in the `/birds/` folder:
```apache
Options -MultiViews
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteRule ^ index.html [QSA,L]
```

**Nginx** — add a `location` block:
```nginx
location /birds/ {
    root /var/www/toddboswell.com;
    try_files $uri $uri/ /birds/index.html;
}
```

Then visit: `https://toddboswell.com/birds/`

---

### Option B — Root of toddboswell.com

Upload the contents of this ZIP to your web root (`public_html/` or `www/`).
Same `.htaccess` or `nginx` rule as above, applied to `/`.

---

### Option C — Netlify / Vercel drag-and-drop (30 seconds)

1. Go to [netlify.com](https://netlify.com) → Sites → drag this folder onto the page
2. Done. You'll get a URL like `https://todd-bird-id.netlify.app`
3. Add a custom domain in Netlify settings to point it at `toddboswell.com/birds`

---

### Option D — GitHub Pages

1. Create a new repo, push the contents of this ZIP as the `main` branch
2. Settings → Pages → Source: `main` / `/ (root)`
3. Add a `CNAME` file containing `toddboswell.com` if using a custom domain

---

### HTTPS requirement

The microphone recording feature requires HTTPS. Most hosts provide this automatically.
If using a subdirectory, the parent domain must already be on HTTPS.

---

### First load note

The BirdNET model (~51 MB) downloads on first use, not on page load.
It caches in the browser after the first download — subsequent uses are instant.

---

### No build step needed

This is a pre-built static site. Just upload the files. No Node.js, no npm, no build tools required on the server.

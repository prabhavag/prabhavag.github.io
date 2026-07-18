---
title: Welcome!
bibliography: bibliography.bib
---

Hi! This is [Prabhav](https://www.linkedin.com/in/prabhav-agrawal-81468315/). I am a Research Engineer in Meta SuperIntelligence team, and this is my notes page. The content is mostly around different aspect of machine learning research and engineering. Hope, you find it interesting!

<div style="text-align: center;">
  <img src="assets/xkcd_ml.png" width="50%">
  <figcaption> Image source: <a href="https://xkcd.com/1838/">XKCD</a>
  </figcaption>
</div>


Finished blog posts:

<style>
.blog-preview-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin: 20px 0;
}

.blog-preview-link {
  text-decoration: none;
  color: inherit;
  display: block;
  width: 100%;
  height: 100%;
  position: relative;
  z-index: 1;
}

.blog-preview {
  border: 1px solid #e5e5e5;
  border-radius: 12px;
  padding: 24px;
  background-color: #faf8f8;
  transition: all 0.3s ease;
  max-width: 100%;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  cursor: pointer;
  position: relative;
  z-index: 1;
}

.blog-preview h3 a[role="anchor"] {
  pointer-events: none !important;
  text-decoration: none !important;
}

.blog-preview:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
  border-color: #284b63;
  z-index: 10;
  background-color: #faf8f8;
}

.blog-preview h3 {
  margin: 0 0 4px 0 !important;
  color: #284b63;
  font-size: 1.5em;
  font-weight: 600;
  line-height: 1.2;
  transition: color 0.3s ease;
}

.blog-preview:hover h3 {
  color: #1a3a4a;
}

.blog-preview .meta {
  color: #646464;
  font-size: 0.9em;
  margin-bottom: 8px !important;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: center;
}

.blog-preview .meta span {
  display: flex;
  align-items: center;
  gap: 4px;
}

.blog-preview .excerpt {
  color: #2b2b2b;
  line-height: 1.6;
  margin-bottom: 0;
  font-size: 1em;
}


@media (prefers-color-scheme: dark) {
  .blog-preview {
    background-color: #161618;
    border-color: #393639;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  }
  
  .blog-preview:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    border-color: #7b97aa;
    z-index: 10;
    background-color: #161618;
  }
  
  .blog-preview h3 {
    color: #7b97aa;
  }
  
  .blog-preview:hover h3 {
    color: #9bb3c6;
  }
  
  .blog-preview .meta {
    color: #d4d4d4;
  }
  
  .blog-preview .excerpt {
    color: #ebebec;
  }
  
}

@media (max-width: 768px) {
  .blog-preview {
    padding: 20px;
  }
  
  .blog-preview .meta {
    gap: 12px;
  }
  
  .blog-preview h3 {
    font-size: 1.3em;
  }
}
</style>

{{blog-preview:BlogPosts/VAE.md,BlogPosts/KMeans_Clustering.md}}

## Research wikis

Interlinked notes I build up as I read papers — entities, concepts, and source summaries,
cross-referenced.

- [Audio LLM Wiki](wiki/mm-llm-wiki/) — Moshi, Voxtral, Qwen3-Omni, and
  Thinking Machines' interaction models.
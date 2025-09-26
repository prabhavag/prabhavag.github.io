import { QuartzTransformerPlugin } from "../types"
import { PluggableList } from "unified"
import { ReplaceFunction, findAndReplace as mdastFindReplace } from "mdast-util-find-and-replace"
import { Root } from "mdast"
import { VFile } from "vfile"
import readingTime from "reading-time"
import { read } from "to-vfile"
import { FilePath } from "../util/path"
import matter from "gray-matter"

export interface BlogPreviewOptions {
  /**
   * Maximum length for excerpt
   */
  excerptLength: number
}

const defaultOptions: BlogPreviewOptions = {
  excerptLength: 300,
}

// Cache for blog post data
const blogPostCache = new Map<string, any>()

async function getBlogPostData(path: string, excerptLength: number) {
  if (blogPostCache.has(path)) {
    return blogPostCache.get(path)
  }

  try {
    const filePath = `content/${path}` as FilePath
    const fileContent = await read(filePath)
    const { data: frontmatter, content } = matter(fileContent.value.toString())
    
    // Calculate reading time
    const { minutes } = readingTime(content)
    const readingTimeText = `${Math.ceil(minutes)} min read`

    // Extract excerpt from content
    let excerpt = frontmatter.description || ""
    if (!excerpt) {
      // Fallback: extract first few sentences from content
      // Clean up the content by removing markdown syntax but preserving math and special symbols
      let cleanContent = content
        .replace(/^#+\s+/gm, '') // Remove heading markers
        .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markers
        .replace(/\*(.*?)\*/g, '$1') // Remove italic markers
        .replace(/`(.*?)`/g, '$1') // Remove code markers
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links, keep text
        .replace(/\[@\w+\]/g, '') // Remove citations like [@wikipedia_kmeans]
        .replace(/<[^>]+>/g, '') // Remove HTML tags
        .replace(/\s+/g, ' ') // Normalize whitespace
        .trim()
      
      // Split into sentences more intelligently
      const sentences = cleanContent
        .split(/[.!?]+\s+/)
        .filter(s => s.length > 10) // Filter out very short fragments
        .map(s => s.trim())
      
      const finalDesc: string[] = []
      let currentLength = 0
      
      for (const sentence of sentences) {
        if (currentLength + sentence.length > excerptLength) break
        const sentenceWithPeriod = sentence.endsWith('.') ? sentence : sentence + '.'
        finalDesc.push(sentenceWithPeriod)
        currentLength += sentenceWithPeriod.length
      }
      
      excerpt = finalDesc.join(' ')
      
      // If still no excerpt, take first few words
      if (!excerpt || excerpt.length < 20) {
        const words = cleanContent.split(/\s+/).slice(0, 20)
        excerpt = words.join(' ') + (words.length >= 20 ? '...' : '')
      }
    }



    // Format date
    let dateText = "Recent"
    if (frontmatter.date) {
      const date = new Date(frontmatter.date)
      dateText = date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric' 
      })
    }

    // Format tags
    const tags = frontmatter.tags || []
    const tagsText = tags.join(", ")

    // Get title from frontmatter or filename
    const title = frontmatter.title || 
      path.replace('.md', '').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())

    const data = {
      title,
      dateText,
      readingTimeText,
      tagsText,
      excerpt
    }

    blogPostCache.set(path, data)
    return data
  } catch (error) {
    console.warn(`Could not read blog post: ${path}`, error)
    const title = path.replace('.md', '').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    return {
      title,
      dateText: "Recent",
      readingTimeText: "1 min read",
      tagsText: "",
      excerpt: "Click to read more..."
    }
  }
}

export const BlogPreview: QuartzTransformerPlugin<Partial<BlogPreviewOptions>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "BlogPreview",
    markdownPlugins() {
      return [
        () => {
          return async (tree: Root, file: VFile) => {
            const replacements: [RegExp, ReplaceFunction][] = []

            // Match blog preview syntax: {{blog-preview:path1,path2,path3}}
            const blogPreviewRegex = /\{\{blog-preview:(.*?)\}\}/g

            // Find all blog preview matches and process them
            const matches = Array.from(file.value.toString().matchAll(blogPreviewRegex))
            
            for (const match of matches) {
              const pathsString = match[1]
              const paths = pathsString.split(',').map(p => p.trim())
              
              // Generate markdown for blog previews that will be processed through the normal pipeline
              let markdown = '<div class="blog-preview-container">\n\n'
              
              for (const path of paths) {
                const data = await getBlogPostData(path, opts.excerptLength)
                
                markdown += `<a href="${path}" class="blog-preview-link">\n`
                markdown += `  <div class="blog-preview">\n`
                markdown += `    <h3>${data.title}</h3>\n`
                markdown += `    <div class="meta">\n`
                markdown += `      <span>📅 ${data.dateText}</span>\n`
                markdown += `      <span>⏱️ ${data.readingTimeText}</span>\n`
                if (data.tagsText) {
                  markdown += `      <span>🏷️ ${data.tagsText}</span>\n`
                }
                markdown += `    </div>\n`
                markdown += `    <div class="excerpt">\n\n${data.excerpt}\n\n</div>\n`
                markdown += `  </div>\n`
                markdown += `</a>\n\n`
              }
              
              markdown += '</div>'
              
              // Replace the specific match with markdown content
              replacements.push([
                new RegExp(match[0].replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'),
                () => ({
                  type: "html",
                  value: markdown,
                })
              ])
            }

            mdastFindReplace(tree, replacements)
          }
        }
      ]
    }
  }
}
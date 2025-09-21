import rehypeCitation from "@prabhavag/rehype-citation"
import { PluggableList } from "unified"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"

export interface Options {
  bibliographyFile: string
  suppressBibliography: boolean
  linkCitations: boolean
  csl: string
}

const defaultOptions: Options = {
  bibliographyFile: "./bibliography.bib",
  suppressBibliography: false,
  linkCitations: false,
  csl: "apa",
}

export const Citations: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "Citations",
    htmlPlugins(ctx) {
      const plugins: PluggableList = []

      // Add rehype-citation to the list of plugins
      plugins.push([
        rehypeCitation,
        {
          bibliography: opts.bibliographyFile,
          suppressBibliography: opts.suppressBibliography,
          linkCitations: opts.linkCitations,
          csl: opts.csl,
          lang: ctx.cfg.configuration.locale ?? "en-US",
        },
      ])

      // Transform the HTML of the citations; add data-no-popover property to the citation links
      // and modify bibliography formatting to link URLs to titles
      plugins.push(() => {
        return (tree, _file) => {
          visit(tree, "element", (node, _index, _parent) => {
            if (node.tagName === "a" && node.properties?.href?.startsWith("#bib")) {
              // Find the corresponding bibliography entry and get its URL
              const bibId = node.properties.href.substring(1) // Remove the #
              const bibEntry = findBibliographyEntry(tree, bibId)
              if (bibEntry && bibEntry.url) {
                // Replace the internal link with the external URL
                node.properties.href = bibEntry.url
                node.properties.target = "_blank"
                node.properties.rel = "noopener noreferrer"
              } else {
                node.properties["data-no-popover"] = true
              }
            }
            
            // Transform bibliography entries to link URLs to titles and remove all italics
            if (node.tagName === "div" && node.properties?.className?.includes("csl-entry")) {
              let urlNode: any = null
              const fullText = extractTextFromNode(node)
              
              // Find URL node
              visit(node, "element", (childNode) => {
                if (childNode.tagName === "a" && 
                    childNode.properties?.href && 
                    typeof childNode.properties.href === "string" && 
                    childNode.properties.href.startsWith("http")) {
                  urlNode = childNode
                }
              })
              
              // Remove all italics from the bibliography entry
              visit(node, "element", (childNode) => {
                if (childNode.tagName === "i" || childNode.tagName === "em") {
                  // Convert italic nodes to span nodes (remove italic formatting)
                  childNode.tagName = "span"
                  childNode.properties = childNode.properties || {}
                  delete childNode.properties.style
                }
              })
              
              // If we found a URL, try to identify the title and link it
              if (urlNode) {
                const url = urlNode.properties.href
                
                // Try to find the title by looking for patterns in the text
                // For academic papers, the title is usually between the year and the first period
                const yearMatch = fullText.match(/\((\d{4})\)\.\s*([^.]+)/)
                if (yearMatch) {
                  const titleText = yearMatch[2].trim()
                  
                  // Find and replace the title text with a link
                  const titleRegex = new RegExp(titleText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')
                  
                  // Replace the title text in the node
                  visit(node, "text", (textNode, index, parent) => {
                    if (textNode.value && textNode.value.includes(titleText)) {
                      const parts = textNode.value.split(titleRegex)
                      const newChildren = []
                      
                      for (let i = 0; i < parts.length; i++) {
                        if (parts[i]) {
                          newChildren.push({
                            type: "text",
                            value: parts[i]
                          })
                        }
                        if (i < parts.length - 1) {
                          newChildren.push({
                            type: "element",
                            tagName: "a",
                            properties: {
                              href: url,
                              target: "_blank",
                              rel: "noopener noreferrer",
                              className: "title-link"
                            },
                            children: [{ type: "text", value: titleText }]
                          })
                        }
                      }
                      
                      // Replace the text node with the new structure
                      parent.children.splice(index, 1, ...newChildren)
                    }
                  })
                }
                
                // Remove the original URL link
                urlNode.tagName = "span"
                delete urlNode.properties.href
                delete urlNode.properties.target
                delete urlNode.properties.rel
                urlNode.children = []
              }
            }
          })
        }
      })

      return plugins
    },
  }
}

// Helper function to extract text content from a node
function extractTextFromNode(node: any): string {
  if (node.type === "text") {
    return node.value || ""
  }
  
  if (node.children) {
    return node.children.map((child: any) => extractTextFromNode(child)).join("")
  }
  
  return ""
}

// Helper function to find a bibliography entry by ID and extract its URL
function findBibliographyEntry(tree: any, bibId: string): { url: string } | null {
  let foundEntry: any = null
  
  visit(tree, "element", (node) => {
    if (node.tagName === "div" && node.properties?.id === bibId) {
      foundEntry = node
    }
  })
  
  if (foundEntry) {
    // Look for the URL in the bibliography entry
    let url: string | null = null
    
    visit(foundEntry, "element", (childNode) => {
      if (childNode.tagName === "a" && 
          childNode.properties?.href && 
          typeof childNode.properties.href === "string" && 
          childNode.properties.href.startsWith("http")) {
        url = childNode.properties.href
      }
    })
    
    if (url) {
      return { url }
    }
  }
  
  return null
}

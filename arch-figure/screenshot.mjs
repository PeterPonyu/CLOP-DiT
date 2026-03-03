// screenshot.mjs — Take high-res screenshot of architecture figure
import puppeteer from 'puppeteer'
import { fileURLToPath } from 'url'
import path from 'path'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

async function takeScreenshot() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  })

  const page = await browser.newPage()

  // Set viewport to 4K for high resolution
  await page.setViewport({
    width: 2640,
    height: 1160,
    deviceScaleFactor: 2  // 2x for retina-quality (effective 5280 x 2320)
  })

  await page.goto('http://localhost:3333', { waitUntil: 'networkidle0', timeout: 30000 })

  // Wait for SVG to render
  await page.waitForSelector('svg', { timeout: 10000 })
  await new Promise(r => setTimeout(r, 2000))  // Extra time for fonts

  // Get the SVG element bounding box
  const svgElement = await page.$('svg')
  const bbox = await svgElement.boundingBox()

  // Screenshot just the SVG with padding
  const padding = 20
  const outPath = path.resolve(__dirname, '..', 'figures', 'v5_publication', 'fig1_architecture.png')

  await page.screenshot({
    path: outPath,
    clip: {
      x: Math.max(0, bbox.x - padding),
      y: Math.max(0, bbox.y - padding),
      width: bbox.width + 2 * padding,
      height: bbox.height + 2 * padding
    },
    type: 'png',
    omitBackground: false
  })

  console.log(`Screenshot saved to: ${outPath}`)
  console.log(`Resolution: ${Math.round((bbox.width + 2*padding) * 2)} x ${Math.round((bbox.height + 2*padding) * 2)} px (2x scale)`)

  // Also save a PDF version
  const pdfPath = path.resolve(__dirname, '..', 'figures', 'v5_publication', 'fig1_architecture.pdf')
  await page.pdf({
    path: pdfPath,
    width: `${bbox.width + 2 * padding}px`,
    height: `${bbox.height + 2 * padding}px`,
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 }
  })
  console.log(`PDF saved to: ${pdfPath}`)

  await browser.close()
}

takeScreenshot().catch(err => {
  console.error('Screenshot failed:', err)
  process.exit(1)
})

export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  ssr: false,
  app: {
    head: {
      link: [
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap' }
      ]
    }
  }
})

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#070A11',
        workspace: '#0A0E1A',
        surface: {
          50: '#1e293b',
          100: '#141B2D',
          200: '#0F1523',
          300: '#0B0F19',
          400: '#080C14',
        },
        border: {
          subtle: '#182235',
          DEFAULT: '#22304A',
          light: '#334366',
        },
        brand: {
          50: '#eff6ff',
          100: '#dbeafe',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        accent: {
          DEFAULT: '#3b82f6',
          hover: '#2563eb',
          cyan: '#06b6d4',
          indigo: '#6366f1',
          emerald: '#10b981',
          amber: '#f59e0b',
          rose: '#f43f5e',
        }
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'sans-serif'
        ],
        mono: [
          'JetBrains Mono',
          'Fira Code',
          'SFMono-Regular',
          'Consolas',
          'monospace'
        ]
      },
      boxShadow: {
        'glow-sm': '0 0 10px -2px rgba(59, 130, 246, 0.25)',
        'glow-md': '0 0 20px -3px rgba(59, 130, 246, 0.35)',
        'glow-emerald': '0 0 15px -3px rgba(16, 185, 129, 0.3)',
        'glow-rose': '0 0 15px -3px rgba(244, 63, 94, 0.3)',
      },
      backgroundImage: {
        'card-gradient': 'linear-gradient(180deg, rgba(15, 21, 35, 0.85) 0%, rgba(10, 14, 26, 0.95) 100%)',
        'subtle-gradient': 'linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.3) 100%)',
        'hero-radial': 'radial-gradient(ellipse at 50% 0%, rgba(59, 130, 246, 0.15) 0%, rgba(7, 10, 17, 0) 70%)',
      }
    },
  },
  plugins: [],
}

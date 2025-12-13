# Styling Quick Start Guide

## 🎨 What Changed?

Your Censorium frontend has been completely restyled with a modern dark theme inspired by the design images you provided.

## ✅ Summary of Changes

### Visual Design
- ✨ **Dark theme** with sophisticated brown-black color palette
- 🎯 **Sidebar navigation** with collapsible menu
- 🔤 **Custom Cursor Gothic fonts** for professional typography
- 🎴 **Card-based layouts** with improved visual hierarchy
- 🌊 **Smooth transitions** and hover effects

### Color Palette
```
Background:   #13120a (deep brown-black)
Cards/Tiles:  #1b1912 (dark brown)
Accents:      #222019 (subtle brown)
Borders:      #22221e (faint lines)
Text:         #e5e5e5 (light gray)
Interactive:  #4a90e2 (blue)
```

## 🚀 Running the Application

Start the frontend development server:

```bash
cd /Users/midhu1/Projects/censorium/frontend
npm run dev
```

Then open http://localhost:3000 in your browser.

## 📁 Modified Files

1. **`/frontend/app/globals.css`**
   - Added Cursor Gothic font declarations
   - Defined color system with CSS variables
   - Added custom scrollbar styling
   - Enhanced animations and focus states

2. **`/frontend/app/layout.tsx`**
   - Removed Google Fonts dependency
   - Now uses Cursor Gothic from globals.css

3. **`/frontend/app/page.tsx`**
   - Complete layout redesign with sidebar
   - Dark theme implementation
   - Improved header and navigation

4. **`/frontend/components/RedactionViewer.tsx`**
   - Dark theme styling for all states
   - Enhanced interactive elements
   - Better visual feedback

5. **`/frontend/public/fonts/`**
   - Added 4 Cursor Gothic font files (Regular, Bold, Italic, Bold Italic)

## 🎯 Key Features

### Collapsible Sidebar
- Click the arrow button at the bottom of sidebar to collapse/expand
- Collapsed: 80px width (icons only)
- Expanded: 256px width (full navigation)

### Upload Zone
- Dark themed dropzone
- Blue highlight on drag-over
- Large, clear icon and instructions

### Redaction Cards
- Dark background with subtle borders
- Settings panel with mode selection (Blur/Pixelate)
- Confidence threshold slider
- Side-by-side original vs redacted comparison

### API Status
- Green dot: API online
- Red dot: API offline
- Yellow dot: Checking connection
- Located in sidebar footer

## 🎨 Customization

All colors are defined in `/frontend/app/globals.css` using CSS variables:

```css
:root {
  --color-bg-primary: #13120a;
  --color-bg-secondary: #1b1912;
  --color-bg-tertiary: #222019;
  --color-border: #22221e;
  --color-text-primary: #e5e5e5;
  --color-text-secondary: #a0a0a0;
  --color-text-muted: #6b6b6b;
  --color-accent: #4a90e2;
  /* ... more colors ... */
}
```

To change colors, simply modify these variables in `globals.css`.

## 🔍 Testing Checklist

After starting the dev server, verify:

- [ ] Cursor Gothic fonts are loading (check in browser DevTools)
- [ ] Dark theme is applied throughout
- [ ] Sidebar navigation works
- [ ] Sidebar collapses/expands smoothly
- [ ] Upload dropzone responds to drag and drop
- [ ] Redaction viewer shows images correctly
- [ ] Settings panel toggles work
- [ ] Download button functions
- [ ] All hover effects work smoothly

## 📦 Build for Production

To create a production build:

```bash
cd /Users/midhu1/Projects/censorium/frontend
npm run build
```

The build was already tested and completed successfully! ✅

## 🐛 Troubleshooting

### Fonts not loading?
1. Check that fonts are in `/frontend/public/fonts/`
2. Clear browser cache
3. Check browser console for 404 errors

### Colors not showing?
1. Ensure you're using a modern browser (Chrome 88+, Firefox 85+, Safari 14+)
2. Check browser DevTools to see if CSS variables are defined
3. Hard refresh the page (Cmd+Shift+R)

### Layout issues?
1. Clear Next.js cache: `rm -rf .next`
2. Restart dev server
3. Check browser console for errors

## 📚 Documentation

For more details, see:
- **STYLING_CHANGES.md** - Complete technical documentation
- **DESIGN_COMPARISON.md** - Before/after comparison and design decisions

## 🎉 Next Steps

Your frontend is ready to use! The styling is production-ready and fully functional. Simply start the dev server and begin using the application.

If you want to make adjustments:
1. Color changes: Edit CSS variables in `globals.css`
2. Layout changes: Modify `page.tsx` or `RedactionViewer.tsx`
3. Add new components: Follow the same dark theme pattern

---

**Status**: ✅ Complete  
**Version**: 2.0.0  
**Date**: November 24, 2025

Enjoy your newly styled Censorium! 🎨✨


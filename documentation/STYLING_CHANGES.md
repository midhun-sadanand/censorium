# Frontend Styling Changes - Censorium

## Overview
The frontend has been completely restyled with a dark theme inspired by modern, professional UI designs. The new design features a sophisticated color palette, custom typography, and improved layout structure.

## Color Palette

### Applied Colors
- **Primary Background**: `#13120a` - Deep brown-black base
- **Secondary Background**: `#1b1912` - Dark brown for cards/tiles
- **Tertiary Background**: `#222019` - Light accent within tiles
- **Border Color**: `#22221e` - Subtle horizontal lines and borders
- **Text Primary**: `#e5e5e5` - Main text color (light gray)
- **Text Secondary**: `#a0a0a0` - Secondary text color
- **Text Muted**: `#6b6b6b` - Muted/disabled text
- **Accent**: `#4a90e2` - Blue accent for interactive elements
- **Success**: `#4ade80` - Green for success states
- **Warning**: `#fbbf24` - Yellow for warnings
- **Error**: `#ef4444` - Red for errors

## Typography

### Custom Fonts
- **Font Family**: Cursor Gothic
- **Variants**: Regular (400), Bold (700), Italic, Bold Italic
- **Format**: WOFF2 web fonts
- **Location**: `/public/fonts/`

The Cursor Gothic font provides a modern, clean appearance with excellent readability and professional weight distribution.

## Layout Changes

### Sidebar Navigation
- **New Feature**: Collapsible sidebar with navigation items
- **Width**: 256px (expanded) / 80px (collapsed)
- **Features**:
  - Logo area with app branding
  - Navigation menu with Overview and Settings
  - API status indicator
  - Collapse/expand toggle button

### Main Content Area
- **Header**: Fixed header with page title and action buttons
- **Content**: Scrollable main content area
- **Spacing**: Generous padding (32px) for better visual hierarchy

## Component Updates

### 1. `globals.css`
- Added Cursor Gothic font-face declarations
- Defined CSS custom properties for the color system
- Implemented custom scrollbar styling
- Added fade-in animation
- Enhanced focus states and selection styling

### 2. `layout.tsx`
- Removed Google Fonts (Inter) dependency
- Simplified layout to use custom fonts from globals.css
- Maintained metadata for SEO

### 3. `page.tsx` (Main Application)
- **New Layout**: Sidebar + Main content structure
- **Upload Zone**: Dark-themed dropzone with improved visual feedback
- **Info Cards**: Redesigned feature cards with icon containers
- **Header**: New header with page title and action buttons
- **Status Indicator**: API status shown in sidebar footer

### 4. `RedactionViewer.tsx`
- **Card Design**: Dark themed cards with subtle borders
- **Header**: Improved file information display
- **Settings Panel**: Dark themed toggle buttons and sliders
- **Image Display**: Better contrast with dark backgrounds
- **Interactive Elements**: Enhanced hover states and transitions
- **Error Messages**: Styled error display with semi-transparent backgrounds

## Design Principles

### Visual Hierarchy
- Clear distinction between background layers
- Consistent border treatment for separation
- Proper spacing and padding throughout

### Interactive Elements
- Smooth hover transitions
- Clear focus states for accessibility
- Disabled states with reduced opacity

### Accessibility
- High contrast text colors
- Proper focus indicators
- Semantic HTML structure maintained

### Responsiveness
- Maintained responsive grid layouts
- Sidebar collapses on smaller screens
- Mobile-friendly touch targets

## Technical Implementation

### CSS Variables
All colors are defined as CSS custom properties in `:root`, making it easy to:
- Maintain consistent theming
- Make global color changes
- Support potential theme switching in the future

### Inline Styles
Used inline styles with CSS variables for dynamic theming:
```tsx
style={{ background: 'var(--color-bg-secondary)' }}
```

This approach provides:
- Type safety with TypeScript
- Dynamic theme support
- Easy maintenance

### Tailwind CSS v4
- CSS-based configuration via `@theme inline`
- No separate config file needed
- Utility classes for layout and spacing

## Browser Compatibility

The styling uses modern CSS features:
- CSS Custom Properties (CSS Variables)
- WOFF2 fonts
- Flexbox and Grid layouts
- Modern selectors (`:focus-visible`, `::selection`)

Supports all modern browsers:
- Chrome/Edge 88+
- Firefox 85+
- Safari 14+

## Future Enhancements

Potential improvements:
- Add theme switcher (light/dark mode toggle)
- Implement color scheme preferences
- Add more animation/transition effects
- Enhanced loading states
- Skeleton loaders for better perceived performance

## Testing

To verify the changes:
1. Start the frontend: `./start_frontend.sh` or `npm run dev`
2. Open http://localhost:3000
3. Check:
   - Fonts are loading correctly
   - Colors match the design specification
   - Sidebar navigation works
   - All interactive elements respond properly
   - Upload and redaction features function normally

## Files Modified

1. `/frontend/app/globals.css` - Complete styling overhaul
2. `/frontend/app/layout.tsx` - Font configuration
3. `/frontend/app/page.tsx` - Layout and design updates
4. `/frontend/components/RedactionViewer.tsx` - Dark theme implementation
5. `/frontend/public/fonts/` - Added Cursor Gothic font files

---

**Last Updated**: November 24, 2025
**Version**: 2.0.0


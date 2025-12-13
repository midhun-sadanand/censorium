'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Settings from '@geist-ui/icons/settings';
import Upload from '@geist-ui/icons/upload';

interface SidebarProps {
  apiStatus?: 'checking' | 'online' | 'offline';
}

export default function Sidebar({ apiStatus }: SidebarProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const pathname = usePathname();

  return (
    <aside 
      className={`transition-all duration-300 ${
        sidebarOpen ? 'w-64' : 'w-20'
      }`}
      style={{ 
        background: 'var(--color-bg-secondary)',
        borderRight: '1px solid var(--color-border)'
      }}
    >
      <div className="flex flex-col h-full">
        {/* Logo */}
        <button 
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-6 flex items-center justify-between w-full transition-opacity hover:opacity-80"
          style={{ borderBottom: '1px solid var(--color-border)' }}
          title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          {sidebarOpen ? (
            <div className="flex items-center gap-3">
              <img 
                src="/censorium.svg" 
                alt="Censorium" 
                className="w-8 h-8"
                style={{ filter: 'brightness(0) saturate(100%) invert(90%)' }}
              />
              <h1 className="text-lg" style={{ color: 'var(--color-text-primary)' }}>Censorium</h1>
            </div>
          ) : (
            <img 
              src="/censorium.svg" 
              alt="Censorium" 
              className="w-8 h-8 mx-auto"
              style={{ filter: 'brightness(0) saturate(100%) invert(90%)' }}
            />
          )}
        </button>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            <Link
              href="/"
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                pathname === '/' ? 'bg-opacity-100' : 'hover:bg-opacity-50'
              }`}
              style={pathname === '/' 
                ? { background: 'var(--color-bg-tertiary)', color: 'var(--color-text-primary)' }
                : { color: 'var(--color-text-secondary)' }
              }
              onMouseEnter={(e) => {
                if (pathname !== '/') {
                  e.currentTarget.style.background = 'var(--color-bg-tertiary)';
                }
              }}
              onMouseLeave={(e) => {
                if (pathname !== '/') {
                  e.currentTarget.style.background = 'transparent';
                }
              }}
            >
              <Upload size={20} />
              {sidebarOpen && <span className="text-sm font-medium">Upload</span>}
            </Link>

            <button 
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors hover:bg-opacity-50"
              style={{ color: 'var(--color-text-secondary)' }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'var(--color-bg-tertiary)'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
            >
              <Settings size={20} />
              {sidebarOpen && <span className="text-sm font-medium">Settings</span>}
            </button>
          </div>
        </nav>

        {/* API Status */}
        {apiStatus && (
          <div className="p-4" style={{ borderTop: '1px solid var(--color-border)' }}>
            <div className="flex items-center gap-2 px-3 py-2">
              <div className={`h-2 w-2 rounded-full ${
                apiStatus === 'online' ? 'bg-green-500' : 
                apiStatus === 'offline' ? 'bg-red-500' : 
                'bg-yellow-500'
              }`} />
              {sidebarOpen && (
                <span className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                  {apiStatus === 'online' ? 'API Online' : 
                   apiStatus === 'offline' ? 'API Offline' : 
                   'Checking...'}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}


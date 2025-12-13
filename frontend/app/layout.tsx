import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Censorium - Image Redaction",
  description: "Real-time face and license plate redaction for privacy protection",
  icons: {
    icon: '/censorium.svg',
    shortcut: '/censorium.svg',
    apple: '/censorium.svg',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

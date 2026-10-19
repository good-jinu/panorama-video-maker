import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Panorama Video Maker',
  description: 'Make a side-scrolling journey video by chatting',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Monitoring And Control Imitation Learning",
  description:
    "Centralized control, 3D simulation, motion analysis, and evaluation platform for MJLab + BeyondMimic + GMR Humanoid Imitation Learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased selection:bg-lime-400/30 selection:text-zinc-900">
        {children}
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Research OS",
  description: "AI-native research workspace for ML papers and systems.",
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

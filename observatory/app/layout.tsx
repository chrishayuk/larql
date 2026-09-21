import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";
export async function generateMetadata(): Promise<Metadata> {
  const incoming = await headers();
  const host = (incoming.get("x-forwarded-host") ?? incoming.get("host") ?? "localhost:3000").split(",")[0].trim();
  const origin = new URL(`${host.startsWith("localhost") ? "http" : "https"}://${host}`);
  return {
  title: "LARQL Observatory — watch computation unfold",
  description: "A HAUSE research instrument for the model as a computational database. Four explicitly synthetic fixture stories, synchronized Map, Trace, Graph, and replay.",
  metadataBase: origin,
  openGraph: { title: "LARQL Observatory", description: "Four synthetic studies. One computational instrument.", images: [{ url: new URL("/og.png", origin).href, width: 1536, height: 1024, alt: "LARQL Observatory — Watch computation unfold. Synthetic UI preview." }] },
  twitter: { card: "summary_large_image", images: [new URL("/og.png", origin).href] },
  };
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" data-mode="dark"><body>{children}</body></html>;
}

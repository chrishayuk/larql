import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  ...(process.env.OBSERVATORY_TARGET === "fly" ? { output: "standalone" as const } : {}),
};

export default nextConfig;

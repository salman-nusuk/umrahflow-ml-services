/* eslint-disable @typescript-eslint/no-require-imports */
import { PrismaClient } from "@prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import bcrypt from "bcryptjs";
import { config } from "dotenv";
import path from "node:path";

config({ path: path.resolve(__dirname, "..", ".env") });

async function main() {
  const email = (process.env.ADMIN_EMAIL ?? "").trim().toLowerCase();
  const password = process.env.ADMIN_PASSWORD ?? "";
  const name = process.env.ADMIN_NAME ?? "Workspace admin";

  if (!email || !email.includes("@")) {
    console.error("ADMIN_EMAIL must be a valid email. Aborting.");
    process.exit(2);
  }
  if (password.length < 8) {
    console.error("ADMIN_PASSWORD must be at least 8 characters. Aborting.");
    process.exit(2);
  }

  const adapter = new PrismaPg({ connectionString: process.env.DATABASE_URL });
  const db = new PrismaClient({ adapter });
  try {
    const existing = await db.user.findUnique({ where: { email } });
    const passwordHash = await bcrypt.hash(password, 12);

    if (existing) {
      await db.user.update({
        where: { id: existing.id },
        data: { passwordHash, role: "ADMIN", active: true, name },
      });
      console.log(`Updated existing user ${email} to ADMIN with new password.`);
    } else {
      const created = await db.user.create({
        data: {
          email,
          name,
          role: "ADMIN",
          passwordHash,
          active: true,
        },
      });
      console.log(`Created admin user ${email} (id=${created.id}).`);
    }
  } finally {
    await db.$disconnect();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

import { PrismaClient } from "@prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import "dotenv/config";

const adapter = new PrismaPg({ connectionString: process.env.DATABASE_URL });
const db = new PrismaClient({ adapter });

async function main() {
  const agents = [
    {
      name: "Al-Haramain Travels",
      crNumber: "CR-2019-4521",
      contactName: "Afaq Ahmad",
      phone: "+923177677524",
      email: "afaq@alharamain.pk",
      city: "Lahore",
      creditLimit: 500000,
    },
    {
      name: "Karachi Travels",
      crNumber: "CR-2020-8812",
      contactName: "Salman Ali",
      phone: "+923001234567",
      email: "salman@karachitravels.pk",
      city: "Karachi",
      creditLimit: 350000,
    },
    {
      name: "Safar-e-Ibadat Nusuk",
      crNumber: "CR-2018-1120",
      contactName: "Sadia Khan",
      phone: "+966538979572",
      email: "ops@safarnusuk.sa",
      city: "Makkah",
      creditLimit: 800000,
    },
  ];

  for (const a of agents) {
    const agent = await db.subAgent.upsert({
      where: { phone: a.phone } as never,
      create: a as never,
      update: a as never,
    }).catch(async () => {
      const existing = await db.subAgent.findFirst({ where: { phone: a.phone } });
      if (existing) return db.subAgent.update({ where: { id: existing.id }, data: a as never });
      return db.subAgent.create({ data: a as never });
    });

    // Two vouchers per agent
    for (let i = 0; i < 2; i++) {
      const ubNum = `UB-${4500 + Math.floor(Math.random() * 500)}`;
      const uidNum = `UR-${Math.floor(Math.random() * 9000 + 1000)}`;
      const total = 80000 + Math.floor(Math.random() * 80000);
      const paid = i === 0 ? total : Math.floor(total * 0.4);
      await db.voucher.create({
        data: {
          ubNumber: ubNum,
          uid: uidNum,
          status: i === 0 ? "SUBMITTED" : "PENDING_REVIEW",
          subAgentId: agent.id,
          sourceType: "SUB_AGENT",
          airline: "PIA",
          flightNumber: `PK-${700 + Math.floor(Math.random() * 60)}`,
          departureCity: a.city,
          arrivalCity: "Madinah",
          departureDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * (20 + i * 7)),
          returnDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * (34 + i * 7)),
          hotelName: "Dar Al Eiman Royal",
          hotelCity: "Madinah",
          totalAmount: total,
          amountReceived: paid,
        },
      });
    }
  }

  // Link any existing conversation to its matching agent
  const conversations = await db.conversation.findMany({ where: { subAgentId: null } });
  for (const c of conversations) {
    const match = await db.subAgent.findFirst({ where: { phone: c.phone } });
    if (match) {
      await db.conversation.update({ where: { id: c.id }, data: { subAgentId: match.id } });
      console.log(`Linked conversation ${c.phone} → ${match.name}`);
    }
  }

  console.log("✓ Seeded demo agents + vouchers");
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => db.$disconnect());

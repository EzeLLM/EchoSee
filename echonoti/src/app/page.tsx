import { getNotifications } from "@/lib/db";
import { HomeClient } from "@/components/HomeClient";

export const dynamic = "force-dynamic";

export default async function Home() {
  const notifications = await getNotifications();
  const unreadNotifications = notifications.filter((n) => !n.read);

  return (
    <HomeClient
      initialNotifications={notifications}
      unreadCount={unreadNotifications.length}
    />
  );
}

import { getNotifications } from "@/lib/db";
import { BookmarkedClient } from "@/components/BookmarkedClient";

export const dynamic = "force-dynamic";

export default async function BookmarkedPage() {
  const notifications = await getNotifications();
  const bookmarkedNotifications = notifications.filter((n) => n.bookmarked);

  return (
    <div className="container mx-auto p-4 md:p-8">
      <BookmarkedClient notifications={bookmarkedNotifications} />
    </div>
  );
}

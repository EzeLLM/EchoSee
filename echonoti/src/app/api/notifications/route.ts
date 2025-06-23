import { NextResponse } from 'next/server';
import { addNotification } from '@/lib/db';

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const { headline, content, summary, type } = body;

    if (!headline || !content || !summary || !type) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const notificationData = {
      headline,
      content,
      summary,
      type,
    };

    const newNotification = await addNotification(notificationData);

    return NextResponse.json(newNotification, { status: 201 });
  } catch (error) {
    console.error('Error creating notification:', error);
    if (error instanceof SyntaxError) {
        return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

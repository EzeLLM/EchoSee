import { NextResponse } from 'next/server';
import { getNotificationById, updateNotification, deleteNotification } from '@/lib/db';

export async function GET(
  _request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const notification = await getNotificationById(params.id);
    if (!notification) {
      return NextResponse.json({ error: 'Not Found' }, { status: 404 });
    }
    return NextResponse.json(notification);
  } catch (error) {
    console.error('Error retrieving notification:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

export async function PUT(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const body = await request.json();
    const updated = await updateNotification(params.id, body);
    if (!updated) {
      return NextResponse.json({ error: 'Not Found' }, { status: 404 });
    }
    return NextResponse.json(updated);
  } catch (error) {
    console.error('Error updating notification:', error);
    if (error instanceof SyntaxError) {
      return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

export async function DELETE(
  _request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const success = await deleteNotification(params.id);
    if (!success) {
      return NextResponse.json({ error: 'Not Found' }, { status: 404 });
    }
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting notification:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

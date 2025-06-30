"use client";

import { useState } from "react";
import { useSettingsContext } from "@/components/SettingsProvider";
import { useT } from "@/hooks/use-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import { clearAllNotifications } from "@/lib/actions";

export default function SettingsPage() {
  const { settings, setSettings } = useSettingsContext();
  const t = useT();
  const [maxQueries, setMaxQueries] = useState(settings.maxQueries);
  const [open, setOpen] = useState(false);

  return (
    <div className="container mx-auto p-4 space-y-6">
      <h1 className="text-3xl font-bold font-headline">{t("settings")}</h1>
      <div className="space-y-4 max-w-xl">
        <div className="space-y-2">
          <Label htmlFor="name">{t("name")}</Label>
          <Input
            id="name"
            value={settings.name}
            onChange={(e) => setSettings({ ...settings, name: e.target.value })}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="info">{t("about_you")}</Label>
          <Textarea
            id="info"
            value={settings.info}
            onChange={(e) => setSettings({ ...settings, info: e.target.value })}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="lang">{t("language")}</Label>
          <Select
            value={settings.language}
            onValueChange={(value) => setSettings({ ...settings, language: value })}
          >
            <SelectTrigger id="lang">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="en">English</SelectItem>
              <SelectItem value="ar">Arabic</SelectItem>
              <SelectItem value="tr">Turkish</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex gap-4 pt-2">
          <form action={clearAllNotifications}>
            <Button variant="destructive">{t("clear_history")}</Button>
          </form>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button type="button" variant="secondary">
                {t("deeper_search")}
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-card border-primary/50">
              <DialogHeader>
                <DialogTitle>{t("max_queries")}</DialogTitle>
                <DialogDescription>
                  {"The more the queries, the higher the accuracy of the deeper search. However this will cost more as it uses more tokens."}
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <Input
                  type="number"
                  min={1}
                  value={maxQueries}
                  onChange={(e) => setMaxQueries(parseInt(e.target.value))}
                />
                <Button
                  onClick={() => {
                    setSettings({ ...settings, maxQueries });
                    setOpen(false);
                  }}
                >
                  {t("save")}
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>
    </div>
  );
}

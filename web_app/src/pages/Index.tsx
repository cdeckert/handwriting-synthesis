import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Download, Type, AlignLeft, AlignCenter, AlignRight } from "lucide-react";
import { toast } from "sonner";

type HandwritingStyle = {
  id: string;
  name: string;
  fontClass: string;
  fontFamily: string;
};

type TextAlignment = "left" | "center" | "right";

type AlignmentOption = {
  value: TextAlignment;
  label: string;
  icon: typeof AlignLeft;
  textAnchor: "start" | "middle" | "end";
  xPosition: number;
};

const handwritingStyles: HandwritingStyle[] = [
  { id: "caveat", name: "Caveat", fontClass: "font-handwriting-caveat", fontFamily: "Caveat" },
  { id: "dancing", name: "Dancing Script", fontClass: "font-handwriting-dancing", fontFamily: "Dancing Script" },
  { id: "pacifico", name: "Pacifico", fontClass: "font-handwriting-pacifico", fontFamily: "Pacifico" },
  { id: "shadows", name: "Shadows Into Light", fontClass: "font-handwriting-shadows", fontFamily: "Shadows Into Light" },
  { id: "marker", name: "Permanent Marker", fontClass: "font-handwriting-marker", fontFamily: "Permanent Marker" },
  { id: "indie", name: "Indie Flower", fontClass: "font-handwriting-indie", fontFamily: "Indie Flower" },
];

const alignmentOptions: AlignmentOption[] = [
  { value: "left", label: "Left", icon: AlignLeft, textAnchor: "start", xPosition: 50 },
  { value: "center", label: "Center", icon: AlignCenter, textAnchor: "middle", xPosition: 400 },
  { value: "right", label: "Right", icon: AlignRight, textAnchor: "end", xPosition: 750 },
];

const Index = () => {
  const [text, setText] = useState("Hello World!\nWelcome to Handwriting Studio");
  const [selectedStyle, setSelectedStyle] = useState<HandwritingStyle>(handwritingStyles[0]);
  const [alignment, setAlignment] = useState<TextAlignment>("left");

  const currentAlignment = alignmentOptions.find(a => a.value === alignment) || alignmentOptions[0];

  const downloadSVG = () => {
    const svgElement = document.getElementById("preview-svg");
    if (!svgElement) {
      toast.error("Preview not available");
      return;
    }

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement("a");
    link.href = url;
    link.download = `handwriting-${Date.now()}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    toast.success("SVG downloaded successfully!");
  };

  const lines = text.split("\n");

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/30">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 mb-2">
            <Type className="w-8 h-8 text-primary" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Handwriting Studio
            </h1>
          </div>
          <p className="text-muted-foreground">Transform your text into beautiful handwriting</p>
        </header>

        <div className="grid lg:grid-cols-2 gap-6 max-w-7xl mx-auto">
          {/* Input Section */}
          <Card className="p-6 space-y-6 shadow-lg">
            <div>
              <label className="block text-sm font-semibold mb-2 text-foreground">
                Your Text
              </label>
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter your text here..."
                className="min-h-[200px] resize-none text-base"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 text-foreground">
                Handwriting Style
              </label>
              <Select
                value={selectedStyle.id}
                onValueChange={(value) => {
                  const style = handwritingStyles.find((s) => s.id === value);
                  if (style) setSelectedStyle(style);
                }}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">{" "}

                  {handwritingStyles.map((style) => (
                    <SelectItem key={style.id} value={style.id} className="cursor-pointer">
                      <span className={style.fontClass}>{style.name}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 text-foreground">
                Text Alignment
              </label>
              <Select
                value={alignment}
                onValueChange={(value) => setAlignment(value as TextAlignment)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">
                  {alignmentOptions.map((option) => {
                    const IconComponent = option.icon;
                    return (
                      <SelectItem key={option.value} value={option.value} className="cursor-pointer">
                        <div className="flex items-center gap-2">
                          <IconComponent className="w-4 h-4" />
                          <span>{option.label}</span>
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={downloadSVG}
              className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
              size="lg"
            >
              <Download className="w-4 h-4 mr-2" />
              Download SVG
            </Button>
          </Card>

          {/* Preview Section */}
          <Card className="p-6 flex items-center justify-center shadow-lg bg-white dark:bg-card">
            <div className="w-full h-full min-h-[400px] flex items-center justify-center">
              <svg
                id="preview-svg"
                viewBox="0 0 800 600"
                className="w-full h-auto max-h-[500px]"
                xmlns="http://www.w3.org/2000/svg"
              >
                <defs>
                  <style>
                    {`
                      @import url('https://fonts.googleapis.com/css2?family=${selectedStyle.fontFamily.replace(" ", "+")}:wght@400;700&display=swap');
                    `}
                  </style>
                </defs>
                <rect width="800" height="600" fill="white" />
                {lines.map((line, index) => (
                  <text
                    key={index}
                    x={currentAlignment.xPosition}
                    y={100 + index * 60}
                    fontSize="40"
                    fontFamily={selectedStyle.fontFamily}
                    fill="#1a1a1a"
                    textAnchor={currentAlignment.textAnchor}
                  >
                    {line}
                  </text>
                ))}
              </svg>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Index;

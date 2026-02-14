'use client';

import { useEffect, useRef, useState } from 'react';
import { Title, Text, Card, Flex, Badge, DonutChart } from '@tremor/react';
import { Defect } from '@/types/yield';
import { Maximize2, Target } from 'lucide-react';

interface Props {
    imageUrl: string;
    defects: Defect[];
}

export default function SEMViewer({ imageUrl, defects }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [imageLoaded, setImageLoaded] = useState(false);

    // Calculate defect statistics
    const defectStats = defects.reduce((acc: Record<string, number>, d) => {
        acc[d.type] = (acc[d.type] || 0) + 1;
        return acc;
    }, {});

    const chartData = Object.entries(defectStats).map(([name, value]) => ({
        name,
        amount: value as number,
    }));

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            setImageLoaded(true);
            // Main image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Overlay defects
            defects.forEach((d) => {
                const x = d.x * canvas.width;
                const y = d.y * canvas.height;
                const w = d.width * canvas.width;
                const h = d.height * canvas.height;

                ctx.strokeStyle = d.type === 'Broken' ? '#f43f5e' : d.type === 'Bridge' ? '#f59e0b' : '#3b82f6';
                ctx.lineWidth = 2;
                ctx.strokeRect(x - w / 2, y - h / 2, w, h);

                // Label
                ctx.fillStyle = ctx.strokeStyle;
                ctx.font = 'bold 10px Inter';
                ctx.fillText(d.type, x - w / 2, y - h / 2 - 5);
            });
        };
    }, [imageUrl, defects, imageLoaded]);

    return (
        <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm overflow-hidden">
            <div className="flex items-center justify-between mb-6">
                <Title className="text-slate-100 flex items-center space-x-2">
                    <Target className="text-rose-500 w-5 h-5" />
                    <span>Metrology Vision Analysis</span>
                </Title>
                <Badge color="rose">AI Detection Active</Badge>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 relative group">
                    <canvas
                        ref={canvasRef}
                        width={800}
                        height={600}
                        className="w-full rounded-lg border border-slate-800 bg-slate-950/50 shadow-2xl"
                    />
                    {!imageLoaded && (
                        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 rounded-lg">
                            <Text className="text-slate-500 animate-pulse">Processing SEM Metrology...</Text>
                        </div>
                    )}
                    <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                        <IconButton icon={Maximize2} className="bg-slate-900/80 border-slate-700" />
                    </div>
                </div>

                <div className="flex flex-col justify-between">
                    <div>
                        <Text className="text-slate-400 font-medium mb-4">Defect Distribution</Text>
                        <DonutChart
                            className="h-44"
                            data={chartData}
                            category="amount"
                            index="name"
                            colors={['rose', 'amber', 'blue', 'emerald']}
                            showAnimation={true}
                            variant="pie"
                        />
                        <div className="mt-6 space-y-2">
                            {chartData.map((item) => (
                                <Flex key={item.name} className="text-xs">
                                    <Text className="text-slate-500">{item.name}</Text>
                                    <Text className="text-slate-300 font-mono font-bold">{item.amount} pts</Text>
                                </Flex>
                            ))}
                        </div>
                    </div>

                    <div className="mt-6 p-4 bg-slate-950/50 rounded-lg border border-slate-800">
                        <Text className="text-[10px] uppercase tracking-widest text-slate-500 font-bold mb-2">Analysis Insight</Text>
                        <Text className="text-xs text-slate-400 italic">
                            Predominant <span className="text-rose-400">Broken</span> defects suggest potential over-etching in the center region.
                        </Text>
                    </div>
                </div>
            </div>
        </Card>
    );
}

// Helper button inside the same file for brevity in this specific implementation
function IconButton({ icon: Icon, className, onClick }: { icon: React.ElementType, className?: string, onClick?: () => void }) {
    return (
        <button onClick={onClick} className={`p-2 rounded-md transition-all ${className}`}>
            <Icon className="w-4 h-4" />
        </button>
    );
}

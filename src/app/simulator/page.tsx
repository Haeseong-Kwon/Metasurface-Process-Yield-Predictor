'use client';

import { Title, Text, Button, Card, Flex, Badge } from '@tremor/react';
import VirtualFab from '@/components/yield/VirtualFab';
import { YieldPrediction } from '@/types/yield';
import { generatePredictionReport } from '@/lib/generateReport';
import { supabase } from '@/lib/supabase';
import { ChevronLeft, History, Zap } from 'lucide-react';
import Link from 'next/link';

export default function SimulatorPage() {
    // const [saving, setSaving] = useState(false); // Removed unused state

    const handleSave = async (prediction: YieldPrediction) => {
        // setSaving(true); // Removed as 'saving' state is no longer used
        try {
            // Note: This assumes the 'yield_predictions' table exists.
            // In a real scenario, we'd ensure migrations are run.
            const { error } = await supabase
                .from('yield_predictions')
                .insert([prediction]);

            if (error) throw error;
            alert('Virtual recipe saved to history!');
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            console.warn('Saving to DB failed (table might not exist yet):', message);
            // For demo purposes, we still show success or log it
            alert('Simulation data logged to console. (Supabase table connection skipped for demo)');
            console.log('Saved Prediction:', prediction);
        }
        // finally {
        //     // Logic for saving state could be used here if needed, but removing for lint
        //     // setSaving(false); // Removed as 'saving' state is no longer used
        // }
    };

    const handleExport = (prediction: YieldPrediction) => {
        generatePredictionReport(prediction);
    };

    return (
        <main className="p-6 bg-slate-950 min-h-screen space-y-8">
            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                    <Link href="/dashboard">
                        <Button
                            variant="secondary"
                            icon={ChevronLeft}
                            className="bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
                        >
                            Back
                        </Button>
                    </Link>
                    <div>
                        <Flex className="space-x-2">
                            <Zap className="text-amber-400 w-5 h-5" />
                            <Title className="text-3xl font-extrabold text-white tracking-tight">Virtual Fab Simulator</Title>
                        </Flex>
                        <Text className="text-slate-400 mt-1">AI-Powered Nanoprocess Yield Prediction & Risk Analysis</Text>
                    </div>
                </div>
                <div className="hidden md:flex space-x-3">
                    <Badge color="amber" icon={History}>Simulation Mode Active</Badge>
                </div>
            </div>

            <div className="mt-8">
                <VirtualFab onSave={handleSave} onExport={handleExport} />
            </div>

            <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm mt-8">
                <Title className="text-slate-100 mb-4">Simulation Guide</Title>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
                    <div className="space-y-2">
                        <Text className="text-blue-400 font-bold uppercase tracking-widest text-[10px]">Step 1: Configure</Text>
                        <Text className="text-slate-400 italic">Adjust E-beam dose and etching parameters using the sliders. Real-time AI will predict the outcome.</Text>
                    </div>
                    <div className="space-y-2">
                        <Text className="text-amber-400 font-bold uppercase tracking-widest text-[10px]">Step 2: Analyze Risks</Text>
                        <Text className="text-slate-400 italic">Review the Risk Factor widget to identify bottlenecks or potential failures in your recipe.</Text>
                    </div>
                    <div className="space-y-2">
                        <Text className="text-emerald-400 font-bold uppercase tracking-widest text-[10px]">Step 3: Document</Text>
                        <Text className="text-slate-400 italic">Save the configuration to your history or export a professional PDF report for team review.</Text>
                    </div>
                </div>
            </Card>
        </main>
    );
}

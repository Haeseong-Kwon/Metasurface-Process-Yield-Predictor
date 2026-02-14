'use client';

import { useParams } from 'next/navigation';
import { Title, Text, Card, Grid, Col, Flex, Metric, Badge, Button } from '@tremor/react';
import SEMViewer from '@/components/yield/SEMViewer';
import OptimizationCard from '@/components/yield/OptimizationCard';
import { ProcessWithYield, Defect, OptimizationAdvice } from '@/types/yield';
import Link from 'next/link';
import { FileText, Download, ChevronLeft, Calendar, Database, ShieldAlert, Info } from 'lucide-react';
import { generateProfessionalRunReport } from '@/lib/generateRunReport';
import { useState, useEffect } from 'react';
import { AnalyticsSkeleton } from '@/components/ui/SkeletonLoader';

// Mock generator for a single run detail
const getRunDetail = (id: string): ProcessWithYield => ({
    id,
    created_at: new Date().toISOString(),
    recipe_name: `Batch-${id.split('-').pop()}`,
    etching_time: 145.5,
    pressure: 55,
    temperature: 24.2,
    gas_flow_rate: 105,
    yield_results: [{
        id: 'res-001',
        process_run_id: id,
        yield_rate: 82.4,
        efficiency: 0.74,
        defect_density: 4.2,
        measured_at: new Date().toISOString(),
    }],
    sem_image_url: 'https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=2070', // Using a tech/circuit image as placeholder
    defects: [
        { id: 'd1', type: 'Broken', x: 0.3, y: 0.4, width: 0.05, height: 0.05 },
        { id: 'd2', type: 'Broken', x: 0.32, y: 0.45, width: 0.04, height: 0.04 },
        { id: 'd3', type: 'Bridge', x: 0.6, y: 0.2, width: 0.06, height: 0.03 },
        { id: 'd4', type: 'Short', x: 0.7, y: 0.8, width: 0.04, height: 0.08 },
        { id: 'd5', type: 'Broken', x: 0.1, y: 0.9, width: 0.05, height: 0.02 },
    ] as Defect[],
    advice: [
        { id: 'a1', category: 'Etching', suggestion: 'Reduce etching time by 10s to prevent pattern breakdown (Broken defects).', impact_percent: 4.5, priority: 'High' },
        { id: 'a2', category: 'Pressure', suggestion: 'Increase chamber pressure by 5 mTorr for better plasma uniformity.', impact_percent: 1.2, priority: 'Medium' },
    ] as OptimizationAdvice[]
});

export default function RunDetailPage() {
    const params = useParams();
    const id = params.id as string;
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate industrial data loading precision
        const timer = setTimeout(() => setLoading(false), 1200);
        return () => clearTimeout(timer);
    }, []);

    const run = getRunDetail(id);

    if (loading) {
        return (
            <main className="p-6 bg-slate-950 min-h-screen">
                <AnalyticsSkeleton />
            </main>
        );
    }

    return (
        <main className="p-6 bg-slate-950 min-h-screen space-y-8 selection:bg-blue-500/30">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-900 pb-8">
                <div className="flex items-center space-x-6">
                    <Link href="/dashboard">
                        <button className="p-2.5 bg-slate-900 border border-slate-800 rounded-xl text-slate-400 hover:text-white hover:border-slate-700 transition-all">
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                    </Link>
                    <div>
                        <Flex className="space-x-3">
                            <Title className="text-3xl font-black text-white tracking-tight uppercase">Metrology Detail: {run.recipe_name}</Title>
                            <Badge color="blue" icon={Database} className="bg-blue-500/10 text-blue-400 border-blue-500/20">Batch ID: {id}</Badge>
                        </Flex>
                        <Flex className="mt-1.5 space-x-5">
                            <div className="flex items-center text-slate-500 text-xs font-bold tracking-widest uppercase">
                                <Calendar className="w-3.5 h-3.5 mr-1.5 text-slate-600" /> {new Date(run.created_at).toLocaleDateString()}
                            </div>
                            <div className="flex items-center text-slate-500 text-xs font-bold tracking-widest uppercase">
                                <ShieldAlert className="w-3.5 h-3.5 mr-1.5 text-slate-600" /> Final Yield: <span className="text-emerald-400 ml-1.5">{run.yield_results?.[0]?.yield_rate}%</span>
                            </div>
                        </Flex>
                    </div>
                </div>

                <div className="flex items-center space-x-3">
                    <Button
                        className="bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-300 font-bold"
                        icon={FileText}
                    >
                        Raw Data Logs
                    </Button>
                    <Button
                        className="bg-blue-600 hover:bg-blue-500 border-none shadow-lg shadow-blue-500/10 font-bold"
                        icon={Download}
                        onClick={() => generateProfessionalRunReport(run)}
                    >
                        Download Technical Report
                    </Button>
                </div>
            </div>

            <Grid numItems={1} numItemsLg={3} className="gap-8">
                <Col numColSpanLg={2} className="space-y-8">
                    {/* SEM Viewer Section */}
                    <SEMViewer imageUrl={run.sem_image_url!} defects={run.defects || []} />

                    {/* Detailed Statistics */}
                    <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
                        <Title className="text-slate-100 flex items-center space-x-2 mb-6">
                            <Info className="text-blue-400 w-5 h-5" />
                            <span>Process Parameter Health</span>
                        </Title>
                        <Grid numItems={2} numItemsMd={4} className="gap-6">
                            <div>
                                <Text className="text-slate-500 text-xs uppercase font-bold">Etching Time</Text>
                                <Metric className="text-white text-xl mt-1">{run.etching_time}s</Metric>
                            </div>
                            <div>
                                <Text className="text-slate-500 text-xs uppercase font-bold">Pressure</Text>
                                <Metric className="text-white text-xl mt-1">{run.pressure} mTorr</Metric>
                            </div>
                            <div>
                                <Text className="text-slate-500 text-xs uppercase font-bold">Temperature</Text>
                                <Metric className="text-white text-xl mt-1">{run.temperature}°C</Metric>
                            </div>
                            <div>
                                <Text className="text-slate-500 text-xs uppercase font-bold">Efficiency</Text>
                                <Metric className="text-emerald-400 text-xl mt-1">{run.yield_results?.[0]?.efficiency}</Metric>
                            </div>
                        </Grid>
                    </Card>
                </Col>

                <Col className="space-y-8">
                    {/* AI Advisor Panel */}
                    <OptimizationCard adviceList={run.advice || []} />

                    <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
                        <Title className="text-slate-100 text-sm font-bold uppercase tracking-widest">Metrology History</Title>
                        <div className="mt-4 space-y-4">
                            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between">
                                <Text className="text-xs text-slate-400">Previous Run (ID: 098)</Text>
                                <Badge color="emerald">81.2%</Badge>
                            </div>
                            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between">
                                <Text className="text-xs text-slate-400">Reference Std (V2.1)</Text>
                                <Badge color="blue">85.0%</Badge>
                            </div>
                        </div>
                        <button className="w-full mt-6 py-2 text-xs font-bold text-blue-400 hover:text-blue-300 transition-colors uppercase tracking-widest">
                            View Comparison Report
                        </button>
                    </Card>
                </Col>
            </Grid>
        </main>
    );
}

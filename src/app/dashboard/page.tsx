'use client';

import { useEffect, useState } from 'react';
import { Card, Title, Text, Grid, Flex, Metric, Badge, Button, BadgeDelta } from '@tremor/react';
import { PlusCircle, Database, TrendingUp, Activity, AlertTriangle, Download, FileJson, Zap, Layers } from 'lucide-react';
import RecipeForm from '@/components/dashboard/RecipeForm';
import ProcessAnalytics from '@/components/dashboard/ProcessAnalytics';
import { supabase } from '@/lib/supabase';
import { ProcessRun, YieldResult, ProcessWithYield } from '@/types/yield';
import Link from 'next/link';
import { exportProcessHistoryCSV } from '@/lib/exportData';

// Mock data generator for initial state if DB is empty
const getMockData = (): ProcessWithYield[] => {
    return Array.from({ length: 15 }).map((_, i) => ({
        id: `mock-${i}`,
        created_at: new Date(Date.now() - (15 - i) * 86400000).toISOString(),
        recipe_name: `Batch-${i + 100}`,
        etching_time: 100 + Math.random() * 100,
        pressure: 40 + Math.random() * 20,
        temperature: 20 + Math.random() * 10,
        gas_flow_rate: 90 + Math.random() * 20,
        yield_results: [{
            id: `res-${i}`,
            process_run_id: `mock-${i}`,
            yield_rate: 85 + Math.random() * 10,
            efficiency: 0.7 + Math.random() * 0.25,
            defect_density: 0.5 + Math.random() * 2,
            measured_at: new Date().toISOString(),
        }]
    }));
};

export default function DashboardPage() {
    const [data, setData] = useState<ProcessWithYield[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function fetchData() {
            try {
                const { data: runs, error } = await supabase
                    .from('process_runs')
                    .select('*, yield_results(*)');

                if (error) throw error;

                if (runs && runs.length > 0) {
                    setData(runs as ProcessWithYield[]);
                } else {
                    // If no data in Supabase, use mock data for demo
                    setData(getMockData());
                }
            } catch (error) {
                console.error('Error fetching data:', error);
                setData(getMockData());
            } finally {
                setLoading(false);
            }
        }

        fetchData();
    }, []);

    const avgYield = data.reduce((acc, curr) => acc + (curr.yield_results?.[0]?.yield_rate || 0), 0) / (data.length || 1);
    const avgEfficiency = data.reduce((acc, curr) => acc + (curr.yield_results?.[0]?.efficiency || 0), 0) / (data.length || 1);

    const stats = [
        { title: 'Avg. Yield Rate', metric: `${avgYield.toFixed(1)}%`, delta: '+2.1%', icon: Zap, color: 'emerald' },
        { title: 'Active Batches', metric: '142', delta: '+12', icon: Layers, color: 'blue' },
        { title: 'Process Efficiency', metric: avgEfficiency.toFixed(2), delta: '-0.05', icon: Activity, color: 'amber' },
        { title: 'Anomalies Detected', metric: '2', delta: '0', icon: AlertTriangle, color: 'rose' },
    ];

    if (loading) return (
        <div className="flex items-center justify-center min-h-screen bg-slate-950 text-slate-100">
            <div className="animate-pulse flex flex-col items-center">
                <div className="w-12 h-12 rounded-full bg-blue-500/20 border-2 border-blue-500/50 mb-4 border-t-transparent animate-spin"></div>
                <p className="text-slate-400">Loading Fab Analytics...</p>
            </div>
        </div>
    );

    return (
        <main className="p-6 bg-slate-950 min-h-screen space-y-8 selection:bg-blue-500/30">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-slate-900">
                <div>
                    <Title className="text-3xl font-black text-white tracking-tighter uppercase">Fab Intelligence OS</Title>
                    <Text className="text-slate-500 font-bold uppercase tracking-widest text-xs mt-1">Metasurface Process Control & Yield Optimization</Text>
                </div>
                <div className="flex items-center space-x-3">
                    <Button
                        className="bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-300 font-bold uppercase tracking-wider text-xs"
                        icon={Download}
                        onClick={() => exportProcessHistoryCSV(data)}
                    >
                        Export History
                    </Button>
                    <Link href="/simulator">
                        <Button
                            className="bg-blue-600 hover:bg-blue-500 border-none shadow-lg shadow-blue-500/10 font-bold uppercase tracking-wider text-xs"
                            icon={Activity}
                        >
                            Open Virtual Fab
                        </Button>
                    </Link>
                </div>
            </div>

            <Grid numItems={1} numItemsSm={2} numItemsLg={4} className="gap-6">
                {stats.map((item) => (
                    <Card key={item.title} className="bg-slate-900/50 border-slate-800 backdrop-blur-sm hover:border-slate-700 transition-all cursor-default group">
                        <Flex alignItems="start">
                            <div>
                                <Text className="text-slate-500 text-[10px] uppercase font-bold tracking-widest">{item.title}</Text>
                                <Metric className="text-white mt-1 group-hover:text-blue-400 transition-colors">{item.metric}</Metric>
                            </div>
                            <item.icon className={`w-6 h-6 text-${item.color}-500 opacity-80 group-hover:scale-110 transition-transform`} />
                        </Flex>
                        <div className="mt-4 flex items-center space-x-2">
                            <BadgeDelta deltaType={item.delta.startsWith('+') ? 'moderateIncrease' : 'moderateDecrease'} className="scale-75 origin-left">
                                {item.delta}
                            </BadgeDelta>
                            <Text className="text-[10px] text-slate-600 font-bold uppercase">vs. Last Shift</Text>
                        </div>
                    </Card>
                ))}
            </Grid>

            <div className="mt-8">
                <ProcessAnalytics data={data} />
            </div>

            <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-1">
                    <RecipeForm />
                </div>
                <Card className="lg:col-span-2 bg-slate-900/50 border-slate-800 backdrop-blur-sm overflow-hidden">
                    <Title className="text-slate-100 mb-6 p-6 pb-0 tracking-tight">Recent Process Logs</Title>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead className="border-b border-slate-800">
                                <tr className="text-slate-500 text-xs uppercase tracking-tighter">
                                    <th className="px-4 py-3">Timestamp</th>
                                    <th className="px-4 py-3">Batch Name</th>
                                    <th className="px-4 py-3">Etching (s)</th>
                                    <th className="px-4 py-3">Efficiency</th>
                                    <th className="px-4 py-3">Status</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/50">
                                {data.slice(0, 5).map((run) => (
                                    <tr key={run.id} className="text-slate-300 text-sm hover:bg-slate-800/20 transition-colors">
                                        <td className="px-4 py-4 font-mono text-xs">{new Date(run.created_at).toLocaleTimeString()}</td>
                                        <td className="px-4 py-4 font-medium text-slate-200">{run.recipe_name}</td>
                                        <td className="px-4 py-4">{run.etching_time.toFixed(1)}</td>
                                        <td className="px-4 py-4">{run.yield_results?.[0]?.efficiency.toFixed(3) || 'N/A'}</td>
                                        <td className="px-4 py-4">
                                            <span className="bg-emerald-500/10 text-emerald-500 px-2 py-0.5 rounded text-[10px] font-bold uppercase">Success</span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Card>
            </div>
        </main>
    );
}

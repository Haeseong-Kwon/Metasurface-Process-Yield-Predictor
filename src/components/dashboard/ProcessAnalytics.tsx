'use client';

import { Card, Title, AreaChart, ScatterChart, Text } from '@tremor/react';
import { ProcessWithYield } from '@/types/yield';

interface Props {
    data: ProcessWithYield[];
}

export default function ProcessAnalytics({ data }: Props) {
    // Prepare data for Scatter Chart: Etching Time vs Efficiency
    const scatterData = data.map((d) => ({
        'Etching Time (s)': d.etching_time,
        'Efficiency': d.yield_results?.[0]?.efficiency || 0,
        'Recipe': d.recipe_name,
    }));

    // Prepare data for Lead Time (Yield Trend): Created At vs Yield Rate
    const trendData = data
        .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
        .map((d) => ({
            date: new Date(d.created_at).toLocaleDateString(),
            'Yield Rate (%)': d.yield_results?.[0]?.yield_rate || 0,
        }));

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
                <Title className="text-slate-100">Etching Time vs Efficiency</Title>
                <Text className="text-slate-400 mb-4">Correlation between process duration and result quality</Text>
                <ScatterChart
                    className="h-80 mt-4"
                    data={scatterData}
                    category="Recipe"
                    x="Etching Time (s)"
                    y="Efficiency"
                    colors={['blue', 'emerald', 'amber', 'rose', 'indigo']}
                    showLegend={false}
                />
            </Card>

            <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
                <Title className="text-slate-100">Yield Progress Trend</Title>
                <Text className="text-slate-400 mb-4">Historical yield rate changes over time</Text>
                <AreaChart
                    className="h-80 mt-4"
                    data={trendData}
                    index="date"
                    categories={['Yield Rate (%)']}
                    colors={['emerald']}
                    valueFormatter={(number: number) => `${number.toFixed(1)}%`}
                    showLegend={false}
                    showGridLines={false}
                />
            </Card>
        </div>
    );
}

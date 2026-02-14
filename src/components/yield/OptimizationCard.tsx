'use client';

import { Card, Title, Text, Flex, Badge, Metric } from '@tremor/react';
import { OptimizationAdvice } from '@/types/yield';
import { Lightbulb, TrendingUp, ArrowRight } from 'lucide-react';

interface Props {
    adviceList: OptimizationAdvice[];
}

interface OptimizationButtonProps {
    label: string;
}

function OptimizationButton({ label }: OptimizationButtonProps) {
    return (
        <button className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold rounded-lg transition-all shadow-lg shadow-blue-900/20">
            {label}
        </button>
    );
}

export default function OptimizationCard({ adviceList }: Props) {
    return (
        <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
            <Title className="text-slate-100 flex items-center space-x-2 mb-6">
                <Lightbulb className="text-amber-400 w-5 h-5" />
                <span>AI Optimization Advisor</span>
            </Title>

            <div className="space-y-4">
                {adviceList.map((advice) => (
                    <div
                        key={advice.id}
                        className="p-4 bg-slate-900 border border-slate-800 rounded-xl hover:border-slate-700 transition-all cursor-default group"
                    >
                        <Flex alignItems="start" className="mb-3">
                            <Badge color={advice.priority === 'High' ? 'rose' : 'amber'} className="rounded-full px-3">
                                {advice.priority} Priority
                            </Badge>
                            <div className="flex items-center text-emerald-400 text-xs font-bold uppercase tracking-wider">
                                <TrendingUp className="w-4 h-4 mr-1" />
                                <span>+{advice.impact_percent}% Yield</span>
                            </div>
                        </Flex>

                        <div className="flex items-center space-x-4">
                            <div className="p-3 bg-slate-800 rounded-lg group-hover:bg-blue-500/10 transition-colors">
                                <Text className="text-blue-400 font-bold">{advice.category}</Text>
                            </div>
                            <div className="flex-1">
                                <Text className="text-slate-200 font-medium">{advice.suggestion}</Text>
                                <Text className="text-slate-500 text-xs mt-1">Based on current SEM metrology analysis</Text>
                            </div>
                            <ArrowRight className="text-slate-700 group-hover:text-slate-400 transition-colors" />
                        </div>
                    </div>
                ))}

                {adviceList.length === 0 && (
                    <div className="py-12 border-2 border-dashed border-slate-800 rounded-xl flex flex-col items-center">
                        <Text className="text-slate-500 italic">No critical optimizations suggested for this run.</Text>
                    </div>
                )}
            </div>

            <div className="mt-8 pt-6 border-t border-slate-800">
                <Flex>
                    <div>
                        <Text className="text-slate-500 text-xs uppercase font-bold tracking-widest">Aggregate Improvement</Text>
                        <Metric className="text-emerald-400 text-2xl mt-1">
                            +{adviceList.reduce((acc, curr) => acc + curr.impact_percent, 0).toFixed(1)}%
                        </Metric>
                    </div>
                    <OptimizationButton label="Apply All Fixes" />
                </Flex>
            </div>
        </Card>
    );
}

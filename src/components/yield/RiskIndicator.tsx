'use client';

import { Card, Text, Flex, Title } from '@tremor/react';
import { AlertCircle, CheckCircle, Info } from 'lucide-react';

interface Props {
    riskFactors: string[];
    predictedYield: number;
}

export default function RiskIndicator({ riskFactors, predictedYield }: Props) {
    const isHighRisk = predictedYield < 70 || riskFactors.length > 0;
    const isOptimal = predictedYield >= 90 && riskFactors.length === 0;

    return (
        <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
            <Title className="text-slate-100 flex items-center space-x-2">
                <Info className="w-5 h-5 text-blue-400" />
                <span>Process Risk Analysis</span>
            </Title>

            <div className="mt-4 space-y-4">
                {isOptimal ? (
                    <Flex className="bg-emerald-500/10 p-4 rounded-lg border border-emerald-500/20">
                        <div className="flex items-center space-x-3">
                            <CheckCircle className="text-emerald-500 w-6 h-6" />
                            <div>
                                <Text className="text-emerald-400 font-bold">Optimal Process Window</Text>
                                <Text className="text-emerald-500/70 text-sm">Parameters are within safe operating limits.</Text>
                            </div>
                        </div>
                    </Flex>
                ) : isHighRisk ? (
                    <div className="space-y-3">
                        <Flex className="bg-rose-500/10 p-4 rounded-lg border border-rose-500/20">
                            <div className="flex items-center space-x-3">
                                <AlertCircle className="text-rose-500 w-6 h-6" />
                                <div>
                                    <Text className="text-rose-400 font-bold">Risk Warning</Text>
                                    <Text className="text-rose-500/70 text-sm">Potential yield degradation detected.</Text>
                                </div>
                            </div>
                        </Flex>

                        <ul className="space-y-2">
                            {riskFactors.map((factor, idx) => (
                                <li key={idx} className="flex items-start space-x-2 text-sm text-slate-300">
                                    <span className="text-rose-500 mt-1">•</span>
                                    <span>{factor}</span>
                                </li>
                            ))}
                            {predictedYield < 75 && (
                                <li className="flex items-start space-x-2 text-sm text-slate-300">
                                    <span className="text-rose-500 mt-1">•</span>
                                    <span>Predicted yield falls below performance threshold.</span>
                                </li>
                            )}
                        </ul>
                    </div>
                ) : (
                    <Text className="text-slate-400">Steady state detected. No critical risks identified.</Text>
                )}
            </div>
        </Card>
    );
}

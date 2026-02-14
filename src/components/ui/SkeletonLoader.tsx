export const Skeleton = ({ className }: { className?: string }) => (
    <div className={`animate-pulse bg-slate-800 rounded ${className}`} />
);

export const CardSkeleton = () => (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-4">
        <Skeleton className="h-6 w-1/3" />
        <Skeleton className="h-24 w-full" />
        <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
        </div>
    </div>
);

export const AnalyticsSkeleton = () => (
    <div className="space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
                <CardSkeleton />
            </div>
            <CardSkeleton />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
        </div>
    </div>
);

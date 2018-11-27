export class ClusteringModel {
    constructor(
        public clusters: Cluster[],
    ) { };
}

export class Cluster {
    constructor(
        public elements: ClusterElement[]
    ) { };
}

export class ClusterElement {
    constructor(
        public text: string,
        public type: number
    ) { };
}
